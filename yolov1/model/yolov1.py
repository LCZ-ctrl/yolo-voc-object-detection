import torch
import torch.nn as nn
import numpy as np

from .yolov1_backbone import build_backbone
from .yolov1_neck import build_neck
from .yolov1_head import build_head


class YOLOv1(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 model_name='resnet18', pretrained=True):
        super().__init__()

        # ----------------- Basic Parameters -----------------
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.model_name = model_name
        self.stride = 32

        # ----------------- Model Structure -----------------
        ## backbone network
        self.backbone, feat_dim = build_backbone(self.model_name, pretrained=pretrained)

        ## neck network
        self.neck = build_neck(feat_dim, out_channels=512)
        head_dim = self.neck.out_dim

        ## detection head
        self.head = build_head(head_dim, head_dim, num_classes)

        ## prediction layer
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1, stride=1, padding=0, bias=True)  # [B, 1, H, W]
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)  # [B, C, H, W]
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1, stride=1, padding=0, bias=True)  # [B, 4, H, W]

    def create_grid(self, fmp_size):
        """
        Generate grid coordinate matrix (discrete values)
        """
        # width and height of feature map
        hs, ws = fmp_size

        # generate x and y coordinate of grid
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)], indexing='ij')

        # concat x and y coordinate: [H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        # [H, W, 2] -> [H*W, 2] -> [H*W, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)

        return grid_xy

    def decode_boxes(self, pred_reg, fmp_size):
        """
        Convert txtytwth back to x1y1x2y2
        """
        # generate grid coordinate matrix
        grid_cell = self.create_grid(fmp_size)

        # decode center point coordinates and width and height
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * self.stride
        pred_wh = torch.exp(pred_reg[..., 2:]) * self.stride

        # convert (cx, cy, w, h) to (x1, y1, x2, y2)
        pred_x1y1 = pred_ctr - pred_wh * 0.5  # [B, H*W, 2]
        pred_x2y2 = pred_ctr + pred_wh * 0.5  # [B, H*W, 2]
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)  # [B, H*W, 4]

        return pred_box

    def nms(self, bboxes, scores):
        x1 = bboxes[:, 0]  # xmin
        y1 = bboxes[:, 1]  # ymin
        x2 = bboxes[:, 2]  # xmax
        y2 = bboxes[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            # calculate coordinates of the top-left and bottom-right corners of the intersection
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # calculate width and height of the intersection
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)

            # calculate intersection area
            inter = w * h

            # calculate IoU
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # fliter out bboxes that exceed nms threshold
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, bboxes, scores):
        """
        Filter predictions via confidence threshold and NMS
        """
        # select the best category for each box (the highest score)
        labels = np.argmax(scores, axis=1)

        # choose only the highest category score for each box
        scores = scores[(np.arange(scores.shape[0]), labels)]

        # threshold filtering
        keep = np.where(scores >= self.conf_thresh)[0]
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # nms filtering
        keep = np.zeros(len(bboxes), dtype=np.int64)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)[0]
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        feat = self.backbone(x)
        feat = self.neck(feat)
        cls_feat, reg_feat = self.head(feat)

        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

        # default batch size for testing is 1
        obj_pred = obj_pred[0]  # [H*W, 1]
        cls_pred = cls_pred[0]  # [H*W, C]
        reg_pred = reg_pred[0]  # [H*W, 4]

        # score for each bbox
        scores = obj_pred.sigmoid() * cls_pred.sigmoid()  # [H*W, C]

        # decode bboxes: [H*W, 4]
        bboxes = self.decode_boxes(reg_pred, fmp_size)

        # deploy predictions to CPU for post-processing
        scores = scores.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # post-processing
        bboxes, scores, labels = self.postprocess(bboxes, scores)

        return bboxes, scores, labels

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference(x)
        else:
            feat = self.backbone(x)
            feat = self.neck(feat)
            cls_feat, reg_feat = self.head(feat)

            obj_pred = self.obj_pred(cls_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)

            box_pred = self.decode_boxes(reg_pred, fmp_size)  # [xmin, ymin, xmax, ymax]

            outputs = {
                'pred_obj': obj_pred,  # (torch.Tensor) [B, H*W, 1]
                'pred_cls': cls_pred,  # (torch.Tensor) [B, H*W, C]
                'pred_box': box_pred,  # (torch.Tensor) [B, H*W, 4]
                'stride': self.stride,  # (Int)
                'fmp_size': fmp_size  # (List[int, int]) [H, W]
            }
            return outputs
