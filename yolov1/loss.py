import torch
import torch.nn.functional as F
from matcher import YoloMatcher


def get_ious(bboxes1, bboxes2, iou_type="giou"):
    # basic checks and constants
    eps = 1e-7

    # calculate area for both boxes
    # area = (x2 - x1) * (y2 - y1)
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]).clamp(min=0) * \
            (bboxes1[..., 3] - bboxes1[..., 1]).clamp(min=0)
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]).clamp(min=0) * \
            (bboxes2[..., 3] - bboxes2[..., 1]).clamp(min=0)

    # calculate intersection
    inter_x1 = torch.max(bboxes1[..., 0], bboxes2[..., 0])
    inter_y1 = torch.max(bboxes1[..., 1], bboxes2[..., 1])
    inter_x2 = torch.min(bboxes1[..., 2], bboxes2[..., 2])
    inter_y2 = torch.min(bboxes1[..., 3], bboxes2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # calculate union
    union_area = area1 + area2 - inter_area + eps
    ious = inter_area / union_area

    if iou_type == "iou":
        return ious

    # calculate GIoU extra part
    convex_x1 = torch.min(bboxes1[..., 0], bboxes2[..., 0])
    convex_y1 = torch.min(bboxes1[..., 1], bboxes2[..., 1])
    convex_x2 = torch.max(bboxes1[..., 2], bboxes2[..., 2])
    convex_y2 = torch.max(bboxes1[..., 3], bboxes2[..., 3])

    convex_w = (convex_x2 - convex_x1).clamp(min=0)
    convex_h = (convex_y2 - convex_y1).clamp(min=0)
    convex_area = convex_w * convex_h + eps  # area of smallest enclosing box

    # GIoU formula: IoU - (Convex_Area - Union_Area) / Convex_Area
    gious = ious - (convex_area - union_area) / convex_area

    return gious


class Criterion(object):
    def __init__(self, device, num_classes=20, obj_w=1.0, cls_w=1.0, box_w=5.0):
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = obj_w
        self.loss_cls_weight = cls_w
        self.loss_box_weight = box_w

        # label matching
        self.matcher = YoloMatcher(num_classes=num_classes)

    def loss_objectness(self, pred_obj, gt_obj):
        """
        Compute BCE loss for objectness
        """
        loss_obj = F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')

        return loss_obj

    def loss_classes(self, pred_cls, gt_label):
        """
        Compute BCE loss for classification
        """
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_label, reduction='none')

        return loss_cls

    def loss_bboxes(self, pred_box, gt_box):
        """
        Compute IoU-based regression loss
        """
        ious = get_ious(pred_box, gt_box, iou_type='giou')
        loss_box = 1.0 - ious

        return loss_box

    def __call__(self, outputs, targets, epoch=0):
        device = outputs['pred_cls'][0].device
        stride = outputs['stride']
        fmp_size = outputs['fmp_size']

        # List[B, M, C] -> [B, M, C] -> [B*M, C]
        pred_obj = outputs['pred_obj'].view(-1)  # [B*M,]
        pred_cls = outputs['pred_cls'].view(-1, self.num_classes)  # [B*M, C]
        pred_box = outputs['pred_box'].view(-1, 4)  # [B*M, 4]

        # ------------------ Label Assignment ------------------
        ## match predicted grids with ground truth boxes
        gt_objectness, gt_classes, gt_bboxes = self.matcher(fmp_size=fmp_size, stride=stride, targets=targets)

        gt_objectness = gt_objectness.view(-1).to(device).float()  # [B*M,]
        gt_classes = gt_classes.view(-1, self.num_classes).to(device).float()  # [B*M, C]
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()  # [B*M, 4]

        ## pos_mask: positive sample mask
        pos_masks = (gt_objectness > 0)

        ## num_fgs: positive sample number
        num_fgs = pos_masks.sum().clamp(min=1)

        # ------------------ Loss Calculation ------------------
        ## objectness loss
        loss_obj = self.loss_objectness(pred_obj, gt_objectness)
        loss_obj = loss_obj.sum() / num_fgs

        ## classification loss
        pred_cls_pos = pred_cls[pos_masks]
        gt_classes_pos = gt_classes[pos_masks]
        loss_cls = self.loss_classes(pred_cls_pos, gt_classes_pos)
        loss_cls = loss_cls.sum() / num_fgs

        ## box regression loss
        pred_box_pos = pred_box[pos_masks]
        gt_bboxes_pos = gt_bboxes[pos_masks]
        loss_box = self.loss_bboxes(pred_box_pos, gt_bboxes_pos)
        loss_box = loss_box.sum() / num_fgs

        ## total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        ## save loss in dict
        loss_dict = dict(
            loss_obj=loss_obj,
            loss_cls=loss_cls,
            loss_box=loss_box,
            losses=losses
        )

        return loss_dict


def build_criterion(device, num_classes, obj_w=1.0, cls_w=1.0, box_w=5.0):
    criterion = Criterion(device=device,
                          num_classes=num_classes,
                          obj_w=obj_w,
                          cls_w=cls_w,
                          box_w=box_w)

    return criterion
