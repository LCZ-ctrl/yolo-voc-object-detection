import torch
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)  # 20 classes


class VOCAnnotationTransform(object):
    """
    Transform a VOC annotation XML into a list of bounding boxes and label indices
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES)))
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_idx]

        return res  # [[x1, y1, x2, y2, label_idx], ... ]


class VOCDetection(Dataset):
    def __init__(self, root, image_sets=[('2007', 'trainval'), ('2012', 'trainval')], transform=None,
                 is_train=False):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = VOCAnnotationTransform()
        self.is_train = is_train

        self.ids = []
        for year, split_name in image_sets:
            voc_root = self.root / f'VOC{year}'
            txt_path = voc_root / 'ImageSets' / 'Main' / f'{split_name}.txt'
            with txt_path.open(encoding='utf-8') as f:
                for line in f:
                    img_id = line.strip()
                    if img_id:
                        self.ids.append((voc_root, img_id))

    def load_image_target(self, index):
        """
        Read a single image and its XML annotation
        """
        voc_root, img_id = self.ids[index]

        # image
        img_path = voc_root / 'JPEGImages' / f'{img_id}.jpg'
        image = cv2.imread(str(img_path))
        height, width, channels = image.shape

        # annotation
        xml_path = voc_root / 'Annotations' / f'{img_id}.xml'
        anno = ET.parse(str(xml_path)).getroot()
        if self.target_transform is not None:
            anno = self.target_transform(anno)

        anno = np.array(anno).reshape(-1, 5) if len(anno) > 0 else np.empty((0, 5))
        target = {
            'boxes': anno[:, :4],  # pixel coordinates
            'labels': anno[:, 4].astype(np.long),  # class
            'orig_size': [height, width]  # image size
        }

        return image, target

    def pull_item(self, index):
        """
        Get raw data and apply augmentation
        """
        image, target = self.load_image_target(index)
        image, target, deltas = self.transform(image, target)  # data augmentation
        return image, target, deltas

    def __getitem__(self, index):
        image, target, deltas = self.pull_item(index)
        return image, target, deltas

    def __len__(self):
        return len(self.ids)


class CollateFunc(object):
    def __call__(self, batch):
        targets = []
        images = []

        for sample in batch:
            image = sample[0]
            target = sample[1]

            images.append(image)
            targets.append(target)

        images = torch.stack(images, 0)  # [B, C, H, W]

        return images, targets
