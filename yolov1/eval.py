import numpy as np
import torch


def box_iou_np(box1, box2):
    """
    Compute IoU between two sets of boxes using NumPy
    """
    box1 = np.asarray(box1, dtype=np.float32)
    box2 = np.asarray(box2, dtype=np.float32)

    if box1.size == 0 or box2.size == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)

    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = np.clip(box1[:, 2] - box1[:, 0], a_min=0, a_max=None) * np.clip(
        box1[:, 3] - box1[:, 1], a_min=0, a_max=None
    )
    area2 = np.clip(box2[:, 2] - box2[:, 0], a_min=0, a_max=None) * np.clip(
        box2[:, 3] - box2[:, 1], a_min=0, a_max=None
    )

    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def voc_ap(rec, prec, use_07_metric=False):
    """
    Calculate AP given precisions and recall
    """
    rec = np.asarray(rec)
    prec = np.asarray(prec)

    # 11-point interpolation (VOC 2007)
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
        return ap

    # all-point interpolation
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # integrate the area under the PR curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def nms_np(boxes, scores, iou_thresh=0.5):
    """
    Perform NMS using Numpy
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def postprocess_yolov1(bboxes, scores, conf_thresh=0.01, nms_thresh=0.5):
    """
    Filter predictions by confidence threshold and NMS
    """
    if len(bboxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    labels = np.argmax(scores, axis=1)
    cls_scores = scores[np.arange(scores.shape[0]), labels]

    # threshold filtering
    keep = np.where(cls_scores >= conf_thresh)[0]
    bboxes = bboxes[keep]
    cls_scores = cls_scores[keep]
    labels = labels[keep]

    # nms filtering
    final_keep = []
    for c in np.unique(labels):
        inds = np.where(labels == c)[0]
        c_boxes = bboxes[inds]
        c_scores = cls_scores[inds]

        keep_inds = nms_np(c_boxes, c_scores, iou_thresh=nms_thresh)
        if len(keep_inds) > 0:
            final_keep.extend(inds[keep_inds].tolist())

    if len(final_keep) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    final_keep = np.array(final_keep, dtype=np.int64)
    return bboxes[final_keep], cls_scores[final_keep], labels[final_keep]


@torch.no_grad()
def predict_yolov1_batch(model, images, conf_thresh=0.01, nms_thresh=0.5):
    """
    Run model inference on a batch of images and decode predictions
    """
    device = images.device

    # forward
    feat = model.backbone(images)
    feat = model.neck(feat)
    cls_feat, reg_feat = model.head(feat)

    obj_pred = model.obj_pred(cls_feat)
    cls_pred = model.cls_pred(cls_feat)
    reg_pred = model.reg_pred(reg_feat)
    fmp_size = obj_pred.shape[-2:]

    # reshape predictions [B, C, H, W] -> [B, H*W, C]
    obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)  # [B, H*W, 1]
    cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)  # [B, H*W, C]
    reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)  # [B, H*W, 4]

    # coordinate decoding (grid & offsets)
    grid_cell = model.create_grid(fmp_size)  # [H*W, 2]
    pred_ctr = (torch.sigmoid(reg_pred[..., :2]) + grid_cell) * model.stride
    pred_wh = torch.exp(reg_pred[..., 2:]) * model.stride
    pred_x1y1 = pred_ctr - pred_wh * 0.5
    pred_x2y2 = pred_ctr + pred_wh * 0.5
    pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)  # [B, H*W, 4]

    # confidence score
    scores = obj_pred.sigmoid() * cls_pred.sigmoid()  # [B, H*W, C]

    # post-process each image in batch
    batch_preds = []
    bs = images.shape[0]

    for i in range(bs):
        bboxes = pred_box[i].detach().cpu().numpy()
        sc = scores[i].detach().cpu().numpy()
        boxes, box_scores, labels = postprocess_yolov1(
            bboxes, sc, conf_thresh=conf_thresh, nms_thresh=nms_thresh
        )
        batch_preds.append({
            'boxes': boxes,
            'scores': box_scores,
            'labels': labels
        })

    return batch_preds


def build_gts_from_targets(targets, img_size=None):
    """
    Convert training target format (Tensors) to evaluation format (NumPy)
    """
    gts = []

    for t in targets:
        boxes = t['boxes']
        labels = t['labels']

        if torch.is_tensor(boxes):
            boxes = boxes.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()

        boxes = np.asarray(boxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

        # rescale boxes if normalized
        if img_size is not None and boxes.size > 0 and boxes.max() <= 1.5:
            boxes = boxes * float(img_size)

        gts.append({
            'boxes': boxes,
            'labels': labels
        })

    return gts


def evaluate_map(preds, gts, num_classes=20, iou_thresh=0.5, use_07_metric=False):
    """
    Evaluation through mAP
    """
    assert len(preds) == len(gts), "preds and gts must have same length"

    ap_per_class = {}
    ap_list = []

    for c in range(num_classes):
        gt_by_image = {}
        npos = 0

        for img_id, gt in enumerate(gts):
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']

            if len(gt_boxes) == 0:
                continue

            mask = (gt_labels == c)
            boxes_c = gt_boxes[mask]

            if len(boxes_c) > 0:
                gt_by_image[img_id] = {
                    'boxes': boxes_c.astype(np.float32),
                    'matched': np.zeros(len(boxes_c), dtype=bool)
                }
                npos += len(boxes_c)

        dets = []
        for img_id, pred in enumerate(preds):
            p_boxes = pred['boxes']
            p_scores = pred['scores']
            p_labels = pred['labels']

            if len(p_boxes) == 0:
                continue

            mask = (p_labels == c)
            boxes_c = p_boxes[mask]
            scores_c = p_scores[mask]

            for b, s in zip(boxes_c, scores_c):
                dets.append((img_id, float(s), b.astype(np.float32)))

        if npos == 0:
            ap_per_class[c] = 0.0
            ap_list.append(0.0)
            continue

        if len(dets) == 0:
            ap_per_class[c] = 0.0
            ap_list.append(0.0)
            continue

        dets.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)

        for i, (img_id, score, box) in enumerate(dets):
            if img_id not in gt_by_image:
                fp[i] = 1.0
                continue

            gt_entry = gt_by_image[img_id]
            gt_boxes = gt_entry['boxes']
            matched = gt_entry['matched']

            ious = box_iou_np(box[None, :], gt_boxes).reshape(-1)
            best_idx = np.argmax(ious)
            best_iou = ious[best_idx]

            if best_iou >= iou_thresh and not matched[best_idx]:
                tp[i] = 1.0
                matched[best_idx] = True
            else:
                fp[i] = 1.0

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / np.maximum(npos, 1e-6)
        prec = tp / np.maximum(tp + fp, 1e-6)

        ap = voc_ap(rec, prec, use_07_metric=use_07_metric)
        ap_per_class[c] = float(ap)
        ap_list.append(float(ap))

    mAP = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0
    return mAP, ap_per_class
