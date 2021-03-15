# CV Tasks and Models

## Classification

### AlexNet

- Image Net 우승으로, ML의 패러다임 변화를 이끔.
- LeNet을 계승 및 발전시킴. (ReLU, Dropout 등을 사용함.)

#### VGGNet

- 3x3 크기의 Filter만 사용해, Parameter의 수를 감소 시킴.

#### GoogLeNet

- Inception Block을 통해 다양한 크기의 Filter 사용함.
- Bottleneck 구조(1x1 Filter)를 통해 Parameter의 수를 감소 시킴.
- Auxiliary Classifier를 통해 Degradient Problem을 해결하고자 함.

#### ResNet

- Residual block을 통해 Degradient Problem을 해결하고자 함.
- He Initialization을 통해 초기 값을 낮게 설정함.
- 블록 간 Feature Map의 Channel은 2배, Resolution은 1/2배로 Downsampling
	함.

#### DenseNet

- Residual Block을 Channel 축의 Concatenation으로 결합함.
- Block내 이전 모든 Layer의 Feature Map을 결합함.

#### SENet

- Squeeze와 Excitation 연산으로, 채널 간의 Attention을 계산함.

#### EfficientNet

- Dept, Width, High Resolution을 모두 고려해 효과적으로 설계함.

#### Deformable Convolution

- 물체의 요소를 더 잘 파악하기 위해 Irregular Filter를 제시함.

## Semantic Segmentation

### Fully Convolution Networks (FCN)

- Fully Connected Layer를 1x1 Filter로 대체함.
- Segmentation task의 기본 구조로 사용됨.

### Hyper-columns for object segment

- CN과 유사하나, Region Proposal Algorithm이 사용됨. (End-to-End Model이 아님.)

### U-Net

- Contracting Path는 Downsampling을 수행하며, Receptive Field를 넓힘.
- Expanding Path는 Upsampling을 수행하며, Resolution을 높임.
- 낮은 Layer의 Feature Map을 Concatenation하는 구조를 가짐.

### DeepLab

- CRF를 후처리로 사용해 물체의 경계를 효과적으로 구분함.
- Dilated Convolution을 사용해 효과적으로 Receptive Field를 넓힘.
- Depthwise Separable Convolution으로 연산의 복잡도를 감소 시킴.

## Object Detection

### R-CNN

- 알고리즘 등으로 Region proposal을 받음.(학습이 불가능한 형태임.)
- 제안된 Region을 Warping해, 일정 크기로 변환함.
- 각 Region에 대해 Classification을 수행함.

### Fast R-CNN

- RoI pooling layer을 도입해, 속도를 향상시킴.(CNN Model의 Feature Map을 재활용 함.)
- Output에서 2개의 Branch를 사용해, 두 기능 모두 학습 시킴.(Classification과 Bounding Box Regression)

### Faster R-CNN

- Region Proposal Network (RPN)을 도입해, End-to-End Model을 달성함.

### YOLO

- Image를 grid로 나누어 분류를 수행함.

### SSD

- Multi scale output으로 Bounding Box의 정확도를 향상 시킴.

### RetiaNet (FPN)

- Feature Pyramid Network (FPN)을 제시함.(낮은 Layer의 Feature Map을 Upsampling 시 Sum 함.)
- One Stage의 약점을 극복하고자, Focal loss를 적용함.

### DETR

- Transformer 구조임.
- Backbone CNN의 Feature Map을 Encoder에 입력함.
- Object에 대한 Query를 Decoder에 입력해 결과를 출력함.

### CAM

- Threads hold를 기준으로 Bounding Box도 구성할 수 있음.

### CornerNet

- 좌상단과 우하단 Pixel로 bounding box를 정의함.

### CenterNet

- 중심 Pixel과 높이, 너비를 통해 bounding box를 정의함.
- 중심 Pixel을 통해 Classification의 정확도를 향상 시킴.

## Instance Segmentation

### Mask R-CNN

- Faster R-CNN 구조에 Mask branch를 추가한 형태임.
- RoI Pooling 대신, RoI Align을 사용함.

### YOLACT

- Protonet을 통해 Mask의 Bases를 생성함.

### YolactEdge

- YOLACT의 Layer 2개를 이식해 경량화함.

## Panoptic Segmentation

### UPSNet

- Panoptic Head를 통해 Semantic Head의 결과와 Instance Head의 결과를 종합
	함.

### VPSNet

- Video에 적용하기 위해 Frame 별 Pixel을 Tracking함.

## Landmark Localization

### Hourglass network

- FPN을 stacking 한 구조임.
- Skip Connection에 Hidden Layer를 추가함.

### DensePose

- UV Map을 통해 3D를 표현하고자 함.
- Faster R-CNN에 UV Map을 출력하는 Branch 추가한 형태임.

### RetinaFace

- FPN에 다양한 Task Branch를 추가한 형태임.(최신 모델링 경향을 반영함.)

## Image to Image Translation

### SRGAN

- Super Resolution Task를 수행하는 GAN 모델임. (주어진 Image를 더 높은 해상도의 Image로 생성함.)

### Pix2Pix

- Adversarial Loss와 L1 Loss 사용함.(L1 Loss를 통해 Input의 Content를 잃지 않음.)

### CycleGAN

- Adversarial Loss와 Cycle-consistency Loss 사용함. (Cycle-consistency Loss 를 통해 Input의 Content를 잃지 않음.)

## Multi-modal Tasks

### Image tagging

- Image understanding과 tag generation

### Image captioning

- Image understanding과 sentence generation

### Show, attend, and tell

- Sentence generation 및 word에 따른 attention 출력

### Text-to-image

- Text로부터 Image generation

### Visual question and answering

- Text와 Image로부터 Answer 출력

### Speech-to-Face

- 음성으로부터 Image generation

### Image-to-Speech

- Image로부터 음성 generation

### Sound source localization

- 음성과 Image로부터 특정 Image fixel 추정

## 3D Understanding Tasks

- 3D recognition
- 3D object detection
- 3D semantic segmentation

## Conditional 3D generation

### Mesh R-CNN

- Input은 2D Image이고, Output은 3D Image임.
- Mask R-CNN에 RoI Align을 N개 사용하고, 3D branch를 추가한 구조임.