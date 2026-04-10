import torch
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

from model.yolov2 import YOLOv2
import config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class_names = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
)

VOC_COLOR_MAP = {
    'aeroplane': (0, 110, 210),
    'bicycle': (210, 110, 0),
    'bird': (0, 190, 110),
    'boat': (210, 0, 110),
    'bottle': (110, 190, 0),
    'bus': (0, 190, 190),
    'car': (190, 0, 0),
    'cat': (0, 0, 190),
    'chair': (190, 110, 110),
    'cow': (110, 0, 190),
    'diningtable': (0, 150, 150),
    'dog': (0, 130, 210),
    'horse': (0, 55, 190),
    'motorbike': (150, 90, 210),
    'person': (45, 145, 60),
    'pottedplant': (210, 180, 0),
    'sheep': (170, 160, 210),
    'sofa': (125, 15, 190),
    'train': (35, 35, 135),
    'tvmonitor': (190, 190, 0)
}


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.6):
    """
    Draw bounding box and label text
    """
    h, w, c = img.shape
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # draw bbox rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 4)

    if label is not None:
        # text size
        t_size = cv2.getTextSize(label, 3, fontScale=text_scale, thickness=3)[0]

        tw, th = t_size[0], t_size[1]

        if y1 - th - 5 > 0:
            lx1, ly1 = x1, y1 - th - 5
            lx2, ly2 = x1 + tw, y1
            text_y = y1 - 5
        else:
            lx1, ly1 = x1, y1
            lx2, ly2 = x1 + tw, y1 + th + 5
            text_y = y1 + th + 2

        if lx2 > w:
            diff = lx2 - w + 5
            lx1 -= diff
            lx2 -= diff

        if lx1 < 0:
            lx1 = 5
            lx2 = lx1 + tw

        if ly2 > h:
            ly1 = h - th - 5
            ly2 = h
            text_y = h - 2

        # draw label rectangle
        cv2.rectangle(img, (int(lx1), int(ly1)), (int(lx2), int(ly2)), cls_color, -1)

        # draw text
        cv2.putText(img, label, (int(lx1), int(text_y)), 0,
                    text_scale, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return img


def visualize_results(img, bboxes, scores, labels, vis_thresh, class_colors, class_names):
    """
    Iterate through detection results, filter by threshold, and draw bboxes
    """
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            cls_color = class_colors[cls_id]
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color)
    return img


@torch.no_grad()
def run_inference(image_path, model, device, class_names, class_colors, conf_thresh=0.25):
    """
    Inference: load -> preprocess -> model inference -> postprocess -> visualize
    """
    model.eval()
    # read image
    ori_img = cv2.imread(image_path)
    if ori_img is None: return None
    h, w, c = ori_img.shape

    # preprocess
    img = cv2.resize(ori_img, (config.img_size, config.img_size)).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img /= 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    bboxes, sc, labels = model.inference(img)

    # coordinate restoration
    if len(bboxes) > 0:
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / config.img_size * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / config.img_size * h

        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)

    # draw plot
    vis_img = visualize_results(ori_img, bboxes, sc, labels, conf_thresh, class_colors, class_names)
    return vis_img


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(0)
    class_colors = [VOC_COLOR_MAP[name] for name in class_names]

    model = YOLOv2(device=device, input_size=config.img_size, num_classes=config.num_classes, trainable=False,
                   topk=config.topk, model_name='darknet19', pretrained=config.pretrained).to(device)
    weight_path = Path(config.save_folder) / 'yolov2_voc_best.pth'

    if weight_path.exists():
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f'✅ Loaded weights: {weight_path}')
    else:
        print('❌ No weights found!')

    test_dir = Path(config.root) / 'VOC2007' / 'JPEGImages'
    img_list = list(test_dir.glob('*.jpg'))
    img_path = random.choice(img_list)

    result_img = run_inference(img_path, model, device, class_names, class_colors)

    if result_img is not None:
        save_path = Path('test_images')
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / img_path.name
        cv2.imwrite(save_path, result_img)

        cv2.imshow('Detection Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
