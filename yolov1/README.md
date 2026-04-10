# YOLOv1-VOC

This is a modified **YOLOv1** implementation.

| Model  | Train Dataset                       | Val Dataset  | Epochs | Input Size  | Test Size | mAP@0.5 | mAP@0.6 | mAP@0.75 |
|:-------|:------------------------------------|:-------------|:-------|:------------|:----------|:--------|:--------|:---------|
| YOLOv1 | VOC2007 trainval + VOC2012 trainval | VOC2007 test | 80     | multi-scale | 416x416   | 69.77%  | 61.51%  | 38.92%   |

## Structure

```
├── data/
|   └── VOCdevkit
├── model/
|   ├── __init__.py
|   ├── yolov1_backbone.py
|   ├── yolov1_neck.py
|   ├── yolov1_head.py
|   └── yolov1.py
├── config.py
├── voc.py
├── augmentation.py
├── matcher.py
├── loss.py
├── eval.py
├── train.py
└── test.py
```

<em>Read files in order:</em>
> config.py -> yolov1.py -> voc.py -> augmentation.py -> matcher.py -> loss.py -> eval.py -> train.py -> test.py

## Some Results

<br>
<p align="center">
  <img src="./images/000139.jpg" height="180" />
  <img src="./images/001097.jpg" height="180" />
  <img src="./images/004997.jpg" height="180" />
  <br>
  <img src="./images/004824.jpg" height="180" />
  <img src="./images/006385.jpg" height="180" />
  <img src="./images/003861.jpg" height="180" />
</p>

## Data process

#### <em>Encoding</em>:

File ```voc.py``` handles the loading of the Pascal VOC dataset. The *VOCAnnotationTransform* class parses the category
names from the annotation files. It then determines the category
index based on its position within the predefined VOC_CLASSES list, facilitating the later generation of one-hot class
labels. The *VOCDetection* class is responsible for loading the
images and their corresponding annotations for training and testing.

#### <em>Augmentation</em>:

A series of SSD-style data augmentations are implemented in ```augmentation.py```, including *random cropping*, *random
horizontal flipping*, and *color jittering*, etc. These
augmentations can enrich the diversity of the training data, thereby enhancing the model's robustness and generalization
capability.

#### <em>Ground Truth Matching</em>:

In ```matcher.py```, the *YoloMatcher* class is responsible for label assignment. It maps each ground-truth object to a
specific grid cell on the feature map based on its center
coordinates. This grid cell is then assigned a label of 1, indicating the presence of an object, while all other cells
are labeled 0 . In fact, the original YOLOv1 model uses the Intersection over Union (IoU) between the predicted box and
the ground-truth box as the label for confidence score. Here I simplified it into a binary assignment, only the grid
cell containing the object's center point is considered a positive sample. Specifically, $Pr(\text{objectness}) = 1$
indicates the presence of an object within the cell, while $Pr(\text{objectness}) = 0$ represents the background. For
every assigned positive sample, the matcher generates three supervision targets: a binary objectness flag, a one-hot
category vector, and the ground-truth bounding
box coordinates $(x_{min}, y_{min}, x_{max}, y_{max})$. These targets are then used by the loss function to calculate
the discrepancy between the model's predictions and the actual
annotations.
<br>
<p align="center">
  <img src="./images/iou.png" height="280" />
  <br>
  <em><strong>Intersection over Union (IoU)</strong></em>
</p>

## Model Arichetecture

The simultaneous location and classification of items within an image or video frame is known as **object detection**. A
typical object detection model usually consists of three
components: *backbone network*, *neck network* and *detection head*.

#### <em>Backbone Network</em>:

In this project, I used a *ResNet-18* model as the backbone network.
<br>
<p align="center">
  <img src="./images/resnet18.png" height="260" />
  <br>
  <em><strong>ResNet-18 Model</strong></em>
</p>

ResNet uses operations such as batch normalization and skip connections to help train larger and deeper networks. Here,
I removed the final average pooling layer and the fully connected
classification layer. The maximum downsampling factor of a ResNet-18 network is 32. Therefore, when a 448x448 image is
input, the backbone network will output a feature map with a size of
14x14.  
You can also switch to *ResNet-34*, *ResNet-50* or *ResNet-101* and choose whether to use a pre-trained weight on the
ImageNet in ```train.py```. Theoretically, deeper networks will achieve better performances, but it will also take much
longer time for training correspondingly.

#### <em>Neck Network</em>:

I used a *SPPF* module as the neck network.
<br>
<p align="center">
  <img src="./images/sppf.png" width="666" />
  <br>
  <em><strong>SPPF Module</strong></em>
</p>

First, the input feature map is processed by a 1x1 convolutional layer, compressing its channel to half of its original
value. Then, it is processed three times by a 5x5 max-pooling layer
to obtain features at different receptive field levels. Finally, the original compressed feature map and all pooling
outputs are concatenated along the channel dimension, followed by
another 1×1 convolutional layer to project the features to the specified output dimension.

#### <em>Detection Head</em>:

I used a *Decoupled Head* as the detection head of my YOLOv1 model. It has a very simple structure, outputting two
different features after processed by two 3x3 convolutional layers:
class features $F_{cls}$ and regression features $F_{reg}$.

#### <em>Prediction Layer</em>:

After processed by the former three parts, we get two different features $F_{cls}$ and $F_{reg}$.

- **<em>Bounding Box Confidence Prediction</em>:** I used class features $F_{cls}$ to perform bounding box confidence
  prediction. As said above, unlike original YOLOv1 which uses the IoU
  between the predicted boxes and ground-truth boxes as the optimization target, I simply used binary labels for
  confidence learning. This simplification makes it more accessible and easier
  to implement for beginners. Additionally, I used a Sigmoid function to map the grid confidence scores to the range
  of $[0, 1]$.
- **<em>Classification Confidence Prediction</em>:** I used class features $F_{cls}$ to perform classification
  confidence prediction. So, the class features $F_{cls}$ are used for two
  tasks: object presence detection and category classification. Since the class confidence also within the $[0, 1]$
  range, I applied a Sigmoid function to normalize the output here too.
- **<em>Localization Prediction</em>:** Naturally, I used regression features $F_{reg}$ to perform localization
  prediction. The target range for the bounding box center offsets $(t_x, t_y)$
  is also $[0, 1]$. Consequently, a Sigmoid function is also applied to the network's output for $t_x$ and $t_y$ to
  constrain them within this interval. The remaining two parameters, width
  $w$ and height $h$, are strictly non-negative. Therefore, I used an Exponential function to process them, which
  ensures the outputs remain in the positive real domain while being globally
  differentiable. Here, the Exponential function is scaled by the network's output stride $s$, which implies that the
  predicted $t_w$ and $t_h$ are represented relative to the grid scale.

## Loss Function

The loss function includes confidence loss, classification loss, and bounding box localization loss.

#### <em>Confidence Loss</em>:

Since the confidence output is processed by a Sigmoid function, I used the Binary Cross Entropy (BCE) loss function for
confidence learning. Here, $N_{pos}$ represents the number of
positive samples.

$$L_{\text{conf}} = -\frac{1}{N_{\text{pos}}} \sum_{i=1}^{S^2} \left[ (1 - \hat{C}_i) \log(1 - C_i) + \hat{C}_i \log(C_i) \right]$$

#### <em>Classification Loss</em>:

Similarly, I used the BCE loss function for classification as each category's prediction confidence also been processed
by Sigmoid activation.

$$L_{\text{cls}} = -\frac{1}{N_{\text{pos}}} \sum_{c=1}^{N_c} \sum_{i=1}^{S^2} \mathbb{I}_i^{\text{obj}} \left[ (1 - \hat{p}_{c_i}) \log(1 - p_{c_i}) + \hat{p}_{c_i} \log(p_{c_i}) \right]$$

#### <em>Bounding Box Localization Loss</em>:

For localization, first get the predicted box $B_{\text{pred}}$ from the center point offsets, width, and height. Then
calculate the Generalized IoU (GIoU) between $B_{\text{pred}}$ and
the ground-truth box $B_{\text{gt}}$. Finally, a linear GIoU loss is used to measure the localization error, as shown in
formula below.

$$L_{\text{reg}} = \frac{1}{N_{\text{pos}}} \sum_{i=0}^{S^2} \mathbb{I}_i^{\text{obj}} \left[ 1 - \text{GIoU}(B_{\text{pred}, i}, B_{\text{gt}, i}) \right]$$

#### <em>Total Loss</em>:

By combining formulas above, we can obtain the complete total Loss function. Here, $\lambda_{\text{reg}}$ is the weight
for the localization loss, which is set to 5 by default.

$$L_{\text{loss}} = L_{\text{conf}} + L_{\text{cls}} + \lambda_{\text{reg}} L_{\text{reg}}$$

## Postprocess

#### <em>Compute Detection Scores</em>:

To compute a final detection score, multiply each bounding box's objectness confidence $C_j$ with the maximum category
probability $p_{c_{max}}$. This score represents both the
likelihood of an object existence and the classification accuracy.

$$score_j = C_j \times p_{c_{max}}$$

$$p_{c_{max}} = \max [p(c_1), p(c_2), \dots, p(c_{20})]$$

#### <em>Score Threshold Filtering</em>:

To remove low-quality predictions that likely represent the background, a score threshold (e.g., 0.25) is applied. Any
bounding boxes with a score below this threshold will be discarded.

#### <em>Decoding</em>:

For a grid cell located at $(grid_x, grid_y)$, the model outputs the predicted center point offsets $t_x$ and $t_y$,
along with the log-space transformations for width and height
$t_w$ and $t_h$. To obtain the actual center coordinates $(c_x, c_y)$ and the dimensions $(w, h)$ in the original image
scale, the decoding transformations is employed. A Sigmoid function
($\sigma$) is used to constrain the center offsets within the current grid cell, while an Exponential function ($\exp$)
is used to rescale the width and height relative to the network's
stride.

$$ c_x = [\text{grid}_x + \sigma(t_x)] \times \text{stride} $$

$$ c_y = [\text{grid}_y + \sigma(t_y)] \times \text{stride} $$

$$ w = \exp(t_w) \times \text{stride} $$

$$ h = \exp(t_h) \times \text{stride} $$

#### <em>NMS</em>:

To handle redundant detections for the same object, Non-Maximum Suppression (NMS) is employed. For each category, pick
the highest-scoring box and remove all other overlapping boxes whose
IoU exceeds a nms threshold. This can ensure each object is detected only once.

<br>
<p align="center">
  <img src="./images/002031_test.jpg" width="250" />
  <img src="./images/002031.jpg" width="250"/>
  <br>
  <em><strong>Non-Maximum Suppression (NMS)</strong></em>
</p>

## Train

To start training, run the command -

```
python train.py
```

I used Automatic Mixed Precision (AMP) to accelerate the training process and reduce memory consumption without
sacrificing numerical precision. Furthermore, I used a Cosine Annealing scheduler with a linear warm-up phase during
training. Additionally, Multi-scale Training was implemented, where the input image resolution was randomly sampled
every epoch.

<br>
<p align="center">
  <img src="./images/yolov1_training_metrics.png" height="300" />
  <br>
  <em><strong>Loss and mAP@0.5</strong></em>
</p>

## Test

To test your trained model, run the command -

```
python test.py
```

It will randomly select an image in the test set, and then output the model's prediction results. You can also try your
own images!

<br><br>
<em><strong>My pre-trained
model:</strong></em> [YOLOv1](https://drive.google.com/file/d/1ZqRgSdCnE1isj0LnvndQLmRkCjSWK6MX/view?usp=drive_link)
