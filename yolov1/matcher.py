import torch
import numpy as np


class YoloMatcher(object):
    """
    Assign ground truth objects to grid cells on the feature map
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(self, fmp_size, stride, targets):
        bs = len(targets)
        fmp_h, fmp_w = fmp_size
        gt_objectness = np.zeros([bs, fmp_h, fmp_w, 1])
        gt_classes = np.zeros([bs, fmp_h, fmp_w, self.num_classes])
        gt_bboxes = np.zeros([bs, fmp_h, fmp_w, 4])

        # iterate through each image in the batch
        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            # iterate through the labels of each target in the image
            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                # target bbox coordinate
                x1, y1, x2, y2 = gt_box

                # calculate center point and width and height of the target bbox
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1

                # check if the target bbox is valid
                if bw < 1.0 or bh < 1.0:
                    continue

                # calculate grid coordinate of center point
                xs_c = xc / stride
                ys_c = yc / stride
                grid_x = int(xs_c)
                grid_y = int(ys_c)

                # check if the grid coordinate is valid
                if grid_x < fmp_w and grid_y < fmp_h:
                    # obj
                    gt_objectness[batch_index, grid_y, grid_x] = 1.0

                    # cls
                    cls_ont_hot = np.zeros(self.num_classes)
                    cls_ont_hot[int(gt_label)] = 1.0
                    gt_classes[batch_index, grid_y, grid_x] = cls_ont_hot

                    # bbox
                    gt_bboxes[batch_index, grid_y, grid_x] = np.array([x1, y1, x2, y2])

        # [B, H*W, C]
        gt_objectness = gt_objectness.reshape(bs, -1, 1)
        gt_classes = gt_classes.reshape(bs, -1, self.num_classes)
        gt_bboxes = gt_bboxes.reshape(bs, -1, 4)

        # to tensor
        gt_objectness = torch.from_numpy(gt_objectness).float()
        gt_classes = torch.from_numpy(gt_classes).float()
        gt_bboxes = torch.from_numpy(gt_bboxes).float()

        return gt_objectness, gt_classes, gt_bboxes
