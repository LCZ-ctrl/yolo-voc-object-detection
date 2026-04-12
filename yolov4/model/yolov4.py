import torch
import torch.nn as nn
import numpy as np

from .yolov4_backbone import build_backbone
from .yolov4_neck import build_neck
from .yolov4_pafpn import build_fpn
from .yolov4_head import build_head


class YOLOv4(nn.Module):
    def __init__(self, device, num_classes=20, anchor_size=None, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 topk=100, model_name='cspdarknet53', pretrained=True):
        super().__init__()

        # ----------------- Basic Parameters -----------------
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.model_name = model_name
        self.topk = topk
        self.stride = [8, 16, 32]

        # ----------------- Anchor Box Parameters -----------------
        self.num_levels = 3
        self.num_anchors = len(anchor_size) // self.num_levels
        self.anchor_size = torch.as_tensor(anchor_size).float().view(self.num_levels, self.num_anchors, 2)

        # ----------------- Model Structure -----------------
        ## backbone network
        ## feat_dim = [256, 512, 1024]
        self.backbone, feat_dim = build_backbone(self.model_name, pretrained=pretrained)

        ## neck network: SPP
        self.neck = build_neck(model='csp_sppf', in_channels=feat_dim[-1], out_channels=feat_dim[-1])
        feat_dim[-1] = self.neck.out_dim  # 1024

        ## neck network: FPN
        self.fpn = build_fpn(in_channels=feat_dim, out_channels=int(256 * 1.0))
        self.head_dim = self.fpn.out_channels  # [256, 256, 256]

        ## detection head
        # self.non_shared_heads = nn.ModuleList([
        #     build_head(head_dim, head_dim, num_classes)
        #     for head_dim in self.head_dim
        # ])
        self.non_shared_heads = nn.ModuleList()
        self.cls_dims = []
        self.reg_dims = []

        for head_dim in self.head_dim:
            head = build_head(head_dim, head_dim, num_classes)
            self.non_shared_heads.append(head)
            self.cls_dims.append(head.cls_out_dim)
            self.reg_dims.append(head.reg_out_dim)

        ## prediction layer
        self.obj_preds = nn.ModuleList(
            [nn.Conv2d(dim, 1 * self.num_anchors, kernel_size=1) for dim in self.reg_dims]
        )
        self.cls_preds = nn.ModuleList(
            [nn.Conv2d(dim, self.num_classes * self.num_anchors, kernel_size=1) for dim in self.cls_dims]
        )
        self.reg_preds = nn.ModuleList(
            [nn.Conv2d(dim, 4 * self.num_anchors, kernel_size=1) for dim in self.reg_dims]
        )

        for obj_pred in self.obj_preds:
            nn.init.constant_(obj_pred.bias, -4.0)

        for cls_pred in self.cls_preds:
            nn.init.constant_(cls_pred.bias, -1.0)

    def generate_anchors(self, level, fmp_size):
        fmp_h, fmp_w = fmp_size
        anchor_size = self.anchor_size[level]  # [A, 2]

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)], indexing='ij')
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)  # [H*W, 2]

        # [H*W, 2] -> [H*W, A, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1) + 0.5
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)

        # [A, 2] -> [H*W, A, 2] -> [M, 2]
        anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h * fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)  # [M, 4]

        return anchors

    def decode_boxes(self, level, anchors, reg_pred):
        # decode center point coordinates and width and height
        pred_ctr = (torch.sigmoid(reg_pred[..., :2]) * 3.0 - 1.5 + anchors[..., :2]) * self.stride[level]
        pred_wh = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]

        # convert (cx, cy, w, h) to (x1, y1, x2, y2)
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

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

    def postprocess(self, obj_preds, cls_preds, box_preds):
        all_scores = []
        all_labels = []
        all_bboxes = []

        for obj_pred_i, cls_pred_i, box_pred_i in zip(obj_preds, cls_preds, box_preds):
            # compute scores
            scores_i = (obj_pred_i.sigmoid() * cls_pred_i.sigmoid()).flatten()

            # keep topk predictions according to scores
            k = min(self.topk, scores_i.shape[0])
            topk_scores, topk_idxs = torch.topk(scores_i, k, dim=0)

            # threshold filtering
            keep = topk_scores > self.conf_thresh
            topk_scores = topk_scores[keep]
            topk_idxs = topk_idxs[keep]

            if len(topk_scores) == 0:
                continue

            # get idxs and labels
            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            # pred bbox
            bboxes = box_pred_i[anchor_idxs]

            # deploy predictions to CPU
            all_scores.append(topk_scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        if len(all_scores) == 0:
            return (np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # deploy predictions to CPU
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

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
        # backbone
        pyramid_feats = self.backbone(x)  # [256, 512, 1024]

        # neck
        pyramid_feats[-1] = self.neck(pyramid_feats[-1])

        # fpn
        pyramid_feats = self.fpn(pyramid_feats)  # [256, 256, 256]

        # detection head
        all_anchors = []
        all_obj_preds = []
        all_cls_preds = []
        all_box_preds = []

        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [1, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            fmp_size = cls_pred.shape[-2:]
            anchors = self.generate_anchors(level, fmp_size)  # [M, 4]

            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)  # [M, 1]
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)  # [M, 20]
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)  # [M, 4]

            box_pred = self.decode_boxes(level, anchors, reg_pred)  # [M, 4] (x1, y1, x2, y2)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        # post-process
        bboxes, scores, labels = self.postprocess(all_obj_preds, all_cls_preds, all_box_preds)

        return bboxes, scores, labels

    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            bs = x.shape[0]
            # backbone
            pyramid_feats = self.backbone(x)

            # neck
            pyramid_feats[-1] = self.neck(pyramid_feats[-1])

            # fpn
            pyramid_feats = self.fpn(pyramid_feats)

            # detection head
            all_fmp_sizes = []
            all_obj_preds = []
            all_cls_preds = []
            all_box_preds = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # [B, C, H, W]
                obj_pred = self.obj_preds[level](reg_feat)
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)

                fmp_size = cls_pred.shape[-2:]

                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)

                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

                # decode bbox
                box_pred = self.decode_boxes(level, anchors, reg_pred)

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_fmp_sizes.append(fmp_size)

            # output dict
            outputs = {"pred_obj": all_obj_preds,  # List[torch.Tensor] [[B, M, 1], ...]
                       "pred_cls": all_cls_preds,  # List[torch.Tensor] [[B, M, C], ...]
                       "pred_box": all_box_preds,  # List[torch.Tensor] [[B, M, 4], ...]
                       'fmp_sizes': all_fmp_sizes,  # List[List[int, int]]
                       'strides': self.stride,  # List[int]
                       }

            return outputs