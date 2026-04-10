import torch
import torch.nn as nn
import numpy as np

from .yolov2_backbone import build_backbone
from .yolov2_neck import build_neck
from .yolov2_head import build_head
from config import anchor_size


class YOLOv2(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 topk=100, model_name='darknet19', pretrained=True):
        super().__init__()

        # ----------------- Basic Parameters -----------------
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.model_name = model_name
        self.stride = 32
        self.topk = topk

        # ----------------- Anchor Box Parameters -----------------
        self.anchor_size = torch.as_tensor(anchor_size).view(-1, 2)  # [A, 2]
        self.num_anchors = self.anchor_size.shape[0]  # A

        # ----------------- Model Structure -----------------
        ## backbone network
        self.backbone, feat_dim = build_backbone(self.model_name, pretrained=pretrained)

        ## neck network
        self.neck = build_neck(feat_dim, out_channels=512)
        head_dim = self.neck.out_dim

        ## detection head
        self.head = build_head(head_dim, head_dim, num_classes)

        ## prediction layer
        self.obj_pred = nn.Conv2d(head_dim, 1 * self.num_anchors, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes * self.num_anchors, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4 * self.num_anchors, kernel_size=1)

    def generate_anchors(self, fmp_size):
        """
        Generate grid matrix, each element contains grid coordinates and anchor box scales
        """
        fmp_h, fmp_w = fmp_size

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)], indexing='ij')
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)

        # [H*W, 2] -> [H*W, A, 2] -> [M, 2], M = H*W*A
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)

        # [A, 2] -> [1, A, 2] -> [H*W, A, 2] -> [M, 2]
        anchor_wh = self.anchor_size.unsqueeze(0).repeat(fmp_h * fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)  # [M, 4]

        return anchors

    def create_grid(self, fmp_size):
        fmp_h, fmp_w = fmp_size

        grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)], indexing='ij')
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
        grid_xy = grid_xy.to(self.device)

        return grid_xy

    def decode_boxes(self, anchors, reg_pred):
        """
        Convert txtytwth back to x1y1x2y2
        """
        # decode center point coordinates and width and height
        pred_ctr = (torch.sigmoid(reg_pred[..., :2]) + anchors[..., :2]) * self.stride
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

    def postprocess(self, obj_pred, cls_pred, reg_pred, anchors):
        """
        Filter predictions via confidence threshold and NMS
        """
        # score for each bbox (per class)
        scores = (obj_pred.sigmoid() * cls_pred.sigmoid()).flatten()  # [M*C]

        # keep topk predictions according to scores
        k = min(self.topk, scores.shape[0])
        topk_scores, topk_idxs = torch.topk(scores, k, dim=0)

        # threshold filtering
        keep = topk_scores > self.conf_thresh
        topk_scores = topk_scores[keep]
        topk_idxs = topk_idxs[keep]

        # get idxs and labels
        anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        labels = topk_idxs % self.num_classes

        # get reg_pred and anchors
        reg_pred = reg_pred[anchor_idxs]
        anchors = anchors[anchor_idxs]

        # decode to get pixel coordinates
        bboxes = self.decode_boxes(anchors, reg_pred)

        # deploy predictions to CPU
        scores = topk_scores.cpu().numpy()
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
        bs = x.shape[0]
        feat = self.backbone(x)
        feat = self.neck(feat)
        cls_feat, reg_feat = self.head(feat)

        obj_pred = self.obj_pred(reg_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        # anchors: [M, 4]  (M = H*W*A)
        anchors = self.generate_anchors(fmp_size)

        # [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

        # default batch size for testing is 1
        obj_pred = obj_pred[0]  # [M, 1]
        cls_pred = cls_pred[0]  # [M, C]
        reg_pred = reg_pred[0]  # [M, 4]

        # post-processing
        bboxes, scores, labels = self.postprocess(obj_pred, cls_pred, reg_pred, anchors)

        return bboxes, scores, labels

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference(x)
        else:
            bs = x.shape[0]
            feat = self.backbone(x)
            feat = self.neck(feat)
            cls_feat, reg_feat = self.head(feat)

            obj_pred = self.obj_pred(reg_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            # anchors: [M, 4]
            anchors = self.generate_anchors(fmp_size)

            # [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, 4)

            # decode boxes (broadcast anchors to batch)
            box_pred = self.decode_boxes(anchors, reg_pred)

            outputs = {
                'pred_obj': obj_pred,  # [B, M, 1]
                'pred_cls': cls_pred,  # [B, M, C]
                'pred_box': box_pred,  # [B, M, 4]
                'stride': self.stride,  # 32
                'fmp_size': fmp_size
            }
            return outputs
