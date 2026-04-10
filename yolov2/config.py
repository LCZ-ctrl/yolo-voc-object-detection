import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_size = 416
num_classes = 20
num_workers = 4
batch_size = 16
max_epoch = 80
wp_epoch = 2
pretrained = True

lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
model_name = 'darknet19'

obj_weight = 1.0
cls_weight = 1.0
box_weight = 5.0

root = 'data/VOCdevkit/'
train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
val_sets = [('2007', 'test')]

save_folder = 'weights/'
conf_thresh = 0.01
nms_thresh = 0.5
iou_thresh = 0.5
topk = 100

seed = 42
use_amp = True
anchor_size = [[17, 25], [55, 75], [92, 206], [202, 21], [289, 311]]
multi_scale_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640]
