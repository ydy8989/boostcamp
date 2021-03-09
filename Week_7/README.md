**Week 7 Computer Vision**

 안녕하세요 캠퍼 여러분, 7주차 Computer Vision 강의에 오신 것을 환영합니다.

 

 아래는 Computer Vision 강의에서 다룰 내용 및 과제에 대한 간단한 소개입니다. 

[**Day 31 (3/8 월)**](https://github.com/ydy8989/boostcamp/tree/main/Week_7/Day_1)

1. **Image Classification 1**
	- Computer Vision이란?
	- Computer Vision의 가장 대표적인 task인 Image Classification에 대한 이해 및 대표적인 network 소개
2. **Annotation data efficient learning**
	- 적은 수의 데이터로 효율적으로 학습 하는 방법
	- Data augmentation, Knowledge distillation, Transfer learning, Self-Training...
3. **과제 1**
	- VGG image classification 
		- VGG network의 일부를 구현하고, mask dataset에 대한 classification task를 수행합니다
		- 제출 기한 : 3/8 23:59 

 

[**Day 32 (3/9 화)**](https://github.com/ydy8989/boostcamp/tree/main/Week_7/Day_2)

1. **Image Classification 2**
	- (비교적 최신) Image classification network 소개
2. **Semantic segmentation**
	- 각 pixel이 어떤 종류의 물체인지 pixel-wise classification을 수행하는 semantic segmentation에 대한 이해
	- FCN, U-Net, DeepLab등 대표적인 semantic segmentation architecture 소개
3. **과제 2** 
	- Classification to semantic segmentation
		- 과제 1에서 구현한 VGG network의 FC layer를 1x1 Conv로 바꿔서 segmentation을 수행합니다
		- 제출 기한 : 3/10 23:59

 

[**Day 33 (3/10 수)**](https://github.com/ydy8989/boostcamp/tree/main/Week_7/Day_3)

1. **Object detection**
	- Image 내에서 특정 물체를 detect하고 localize하는 object detection에 대한 이해
	- R-CNN Family, YOLO 및 DETR등 object detection network 소개
2.  **CNN Visualization**
	- CNN 내부의 activation map등을 시각화해 CNN network를 이해하고 debugging하는 방법
	- Gradient Ascent, Saliency map, Grad-CAM 등 시각화 방법 소개
3. **과제 3**
	-  CNN visualization 
		- 강의에서 배운 visualization 방법들을 구현합니다
		- 제출 기한 : 3/12 23:59

 

[**Day 34 (3/11 목)**](https://github.com/ydy8989/boostcamp/tree/main/Week_7/Day_4)

1. **Instance/Panoptic segmentation and landmark localization**
	- 각 pixel이 어떤 instance에 속하는지 pixel-wise classification을 수행하는 instance segmentation, instance+semantic segmentation을 수행하는 panoptic segmantation에 대한 이해
	- Keypoint의 좌표를 예측하는 landmark localization에 대한 이해
	- UPS-Net, VPS-Net, Hourglass등 network에 대한 소개
2. **Conditional Generative Model**
	- Input image를 고려해서 변환된 이미지를 생성할 수 있는 Conditional Generative Model 및 Conditional GAN 소개
	- GAN loss, Pix2Pix, CycleGAN, Perceptual loss 및 적용 예시에 대한 소개
3. **과제 4**
	- Hourglass module 
		- 강의에서 소개된 hourglass module을 구현합니다
		- 제출 기한 : 3/12 23:59

 

[**Day 35 (3/12 금)**](https://github.com/ydy8989/boostcamp/tree/main/Week_7/Day_5)

1. Multimodal captioning and speaking
	- 시각적인 이미지 뿐 아니라 다양한 감각 (텍스트, 오디오 등)을 모두 사용해 학습하는 방법
2. **3D understanding**
	- 3D space를 다룰 수 있는 Computer Vision model에 대한 이해
	- 3D task, 3D dataset 및 Mesh R-CNN 등 3D network 소개
3. **오태현 교수님과의 질의응답 시간**(14:00 - 15:00)