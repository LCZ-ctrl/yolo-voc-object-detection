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


@torch.no_grad()
def predict_yolov4_batch(model, images, conf_thresh=0.001, nms_thresh=0.5):
    """
    Run model inference on a batch of images and decode predictions
    """
    model.eval()

    orig_conf, orig_nms = model.conf_thresh, model.nms_thresh
    model.conf_thresh, model.nms_thresh = conf_thresh, nms_thresh

    bs = images.shape[0]
    batch_preds = []

    for i in range(bs):
        x = images[i: i + 1]
        bboxes, scores, labels = model.inference(x)

        batch_preds.append({
            'boxes': bboxes,  # [N, 4]
            'scores': scores,  # [N]
            'labels': labels  # [N]
        })

    model.conf_thresh, model.nms_thresh = orig_conf, orig_nms

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
