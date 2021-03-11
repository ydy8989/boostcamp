---

layout: post
title: CV / Object detection
subtitle: 객체 인식 분야의 발전사와 advanced skills
thumbnail-img: https://user-images.githubusercontent.com/38639633/110725720-67339280-825b-11eb-8e6d-a272a5a68c7e.png
gh-repo: ydy8989/ydy8989.github.io
gh-badge: [follow]
categories: [BOOSTCAMP]
tags: [boostcamp]
toc: true
comments: true
---



영상 내에 존재하는 객체를 인식하는 방법은 픽셀 마다의 클래스를 분류하는 segmentation 뿐만 아니라 물체 하나하나마다 bounding box 단위의 예측도 있다. 이러한 task를 object detection이라고 하며 자율주행, CCTV 등 다양한 분야에 활용되고 있다. Object detection을 위한 모델은 크게 one-stage detector와 two-stage detector로 구분할 수 있는데 시대의 흐름을 따른 각각의 모델들의 발전사를 소개하기로 한다.



# Object detection

## Traditional methods hand crafted techniques

- 과거에는 사람의 직관에 의한 경계선의 방향 분포로 object를 검출하려는 노력이 있었다.

![image](https://user-images.githubusercontent.com/38639633/110587506-1ff3c600-81b7-11eb-96e3-708126ae2075.png)

- *HOG = histogram of oriented gradients
- a) Average Gradient
- b) max (+) SVM weight
- c) max (-) SVM weight
- d) image
- e) R-HOG descriptor
- f) R-HOG w/(+) SVM
- g) R-HOG w/ (-) SVM

---

**Selective search**

1. over - segmentation(비슷한 색을 가진 것들끼리 영역 분류)
2. iteratively merging similar regions(구간 머징)
3. Extracting candidate boxes from all remaining segmentations(큰 segmentation을 사용)

![image](https://user-images.githubusercontent.com/38639633/110588004-baeca000-81b7-11eb-9803-326fdc23e734.png){:width="60%"}



## R-CNN

2014년 AlexNet의 성공 이후 바로 이를 object detection 분야에 적용한 구현체이다. 

![image](https://user-images.githubusercontent.com/38639633/110588107-ea031180-81b7-11eb-845d-b6f19e30002b.png)

- 2000개 이하로 영역 제안
- input에 적절한 사이즈로 warpping
	- 직사각형이든 무슨형이든 다 같은 사이즈로 input fix시킴
- 영역별로 카테고리를 classification 진행
- 단점은 속도가 매우 느리고, region proposal을 해야만 했기 때문에, 학습 성능 개선 이외에도 한계가 있었다.



## Fast R-CNN

같은 저자들이 속도개선하기 위해 제안한 모델

- Key ideas : 영상 전체에 대해 한번에 추출하고(region 추출이아니라 그냥 이미지 한방에 넣음), 재활용해서 object들을 디텍션한다. 
- CNN에서 미리 convolution feature map까지 뽑아놓는다. (H x W x C shape)
- ROI pooling layer : 한번 뽑아놓은 feature를 여러번 재활용하기 위해 만든 레이어로써, region proposal이 제시한 바운딩 박스를 기준으로 roi에 해당하는 부분의 feature만을 추출하고 resizing한다. 
- 이후는 이를 이용해 분류문제 및 bbox regressor를 진행한다. 

![image](https://user-images.githubusercontent.com/38639633/110588840-ed4acd00-81b8-11eb-90b8-f3a5b4d8bcfe.png)

- 이전 R-CNN보다 18배가 빨라졌지만, 여전히 region proposal 부분은 별도의 알고리즘을 사용하고 있어서 데이터만으로 성능을 높이는 데에는 한계가 있었다. (해당 알고리즘에 따라 정확도가 천차만별이기 때문)



## Faster R-CNN

당연하게도 Fast R-CNN의 region proposal 부분을 성능 개선시켰다. 기존 룰베이스 알고리즘으로 구성된 region proposal을 뉴럴 네트워크 기반으로 구현하였고, object detection 분야의 최초 end-to-end 모델이 되었다. 

구조를 보기 앞서 metric을 한 가지 정의하자

- `IoU(Intersection over Union)` : A metric commonly used in object detection

---

**Anchor boxes**

- A set of pre-defined bounding boxes

- IoU with GT > 0.7 $\Rightarrow$ sample

- IoU with GT < 0.3 $\Rightarrow$ sample

- Feature map 상에서 발생할 것 같은 박스들을 미리 정의해둔 후보군을 `Anchor Box`라 한다. 

	![image](https://user-images.githubusercontent.com/38639633/110630688-09666280-81e9-11eb-8aed-0fe062fd5b5d.png)

- 스케일에 따라, 사각형의 모양에 따라 rough하게 정해둔 박스를 사용한다. 

- faster R-CNN에서는 서로 다른 스케일, 비율별로 9가지의 anchor box를 사용하였다. 

	- 이는 가변적이다. 

- 이 anchor box와 ground truth의 IoU에 따라 `positive sample`과 `negative sample`을 결정한다.

---

가장 큰 핵심적 변화는 selective search에서 Region Proposal network를 제시했다는 점이다. 

![image](https://user-images.githubusercontent.com/38639633/110632350-dd4be100-81ea-11eb-9f93-2a3d83d2a6d4.png){:width="70%"}

- 여러개의 proposal을 제안하게 되고, 해당 box에 대하여 roi pooling을 실시하게되고, classifier 및 regressor를 붙인다. 



**RPN**

feature map 관점에서 fully convolutional하게 sliding window 방식으로(Z모양을 그리며 filter를 입히듯)  매 위치마다 $k$개의 anchor box를 고려한다. 

![image](https://user-images.githubusercontent.com/38639633/110633578-46802400-81ec-11eb-87c2-a5359bb28c0a.png)

각 위치에서 256 dimension의 feature vector하나를 추출하고 2$k$개의 스코어를 뱉는다. 또한, 정교한 $k$개의 위치를 regression하는 regression branch가 따로 있다. 이를 통해 4$k$개의 regression output이 결정된다. 

- 여기서 2$k$와 4$k$인 이유는, object vs Non-object를 앵커마다 고려해야하기 때문에 2$k$이고, regressor의 경우 각 앵커의 좌표 한점과 width, height를 나타내는 4개의 점을 사용해야하기 때문에 4$k$이다. 

그렇다면 왜 앵커박스의 좌표를 왜 다시 regression할까? 

- 만약 앵커박스가 매우 촘촘하면 상관없다. 하지만 계산속도가 어마어마하게 느려진다. 
- 적당한 양의 앵커 박스만을 미리 만들고, 더 정교한 부분은 regression문제로 다시 푼다는 의미이다. 

classification branch는 crossEntropy loss를 사용하고, box coordinate regression문제는 regression loss를 사용하는데, 이 두 loss function이 RPN(region proposal network)을 위한 것이고, 전체 target task를 위한 ROI별 classification에 대한 loss는 따로 또 하나가 추가되어 전체 구조가 End-to-End로 학습되게 되는 원리이다. 

---

테스트 할때에는 rpn에서는 일정 objectness score 이상 나오는 경우도 많고, 엄밀한 threshold를 정하기가 힘들어서 매우많이 중복되거나 bounding box가 생성된다. 

하지만, 이를 효과적으로 필터링, 스크리닝 해주는 방법으로 `Non-Maximum Suppression(NMS)` 알고리즘을 사용하게된다. 이 알고리즘의 순서는 다음과 같다. 매우 고전적으로 행해지던 방식이다. 

1. Select the box with the highest objectiveness score
2. Compare IoU of this box with other boxes
3. Remove the bounding boxes with IoU $\geq$  50%
4. Move to the next highest objectiveness score
5. Repeat `steps 2-4`

![image](https://user-images.githubusercontent.com/38639633/110634746-a9be8600-81ed-11eb-87e4-8909598970af.png){:width="70%"}

---

지금까지 objective detection의 세 가지 방식을 배웠다. 이 외에도 Mask R-CNN 방식이 있으나 이 포스팅에서는 다루지 않는다. 

![image](https://user-images.githubusercontent.com/38639633/110634814-bc38bf80-81ed-11eb-9042-767c24346055.png)



# Single-stage detector

지금까지 two stage detector의 대표주자인 R-CNN family에 대하여 살펴 보았다면, 또 다른 흐름의 one-stage 혹은 single stage detector 방식을 살펴보자 



## Comparison with two-stage detectors

`Single stage detector`의 주 목적은 **정확도를 포기**하더라도 **속도를 매우 빠르게** 하여 real time detection이 가능함에 있다. 이를 위해 Region proposal을 기반으로한 RoI pooling을 사용하지 않고, 곧바로 box regression과 classification을 바로 진행한다. 

![image](https://user-images.githubusercontent.com/38639633/110721694-5c293400-8254-11eb-849b-5b7df186ea46.png)



## You only look once(YOLO)

![image](https://user-images.githubusercontent.com/38639633/110721761-7cf18980-8254-11eb-8dd8-7918d9b001a8.png)

개략적 작동 방식은 다음과 같다. 

- input image를 S by S grid로 나눈다. 
- 각 그리드에 대하여 4개의 좌표, confidence score를 예측하게된다. (Bounding boxes + confidence)
- 이후 그 값에 따른 class score를 따로 예측한다.  (Class probability map)

 

학습 시킬 때에는 이전 R-CNN과 비슷한 방식으로 진행된다. 

- ground box와 match된 anchor box를 positive 로 간주하게 되고, 학습 레이블을 positive로 걸어주며 학습한다. 

![image](https://user-images.githubusercontent.com/38639633/110722150-2b95ca00-8255-11eb-9cb1-e72356bf3246.png)

YOLO의 아키텍쳐는 일반적인 convolutional neural network와 매우 흡사하다.

![image](https://user-images.githubusercontent.com/38639633/110722315-5ed85900-8255-11eb-951e-832e744bdb86.png)

- 차이점이라면, 마지막 output이 7 by 7 map을 30채널로 출력하게 되는데, 이점에 유의하자
- 논문에서는 위 그림을 기준으로 bounding box는 2개(즉, B=2)를 사용하였고, class는 20개(즉, C=20)를 고려하였다. 
- bounding box regressor를 위한 x,y,w,h,obj score 5개에 값이 필요하고, Class probability length까지 총 5B+C의 length를 필요로 한다. 
- 논문에서는 B=2, C=20이므로 총 30 length가 필요하게 되고, 이를 30개의 채널에 대응시켜 각 위치를 mapping하였다. 
- 앞서 개략적인 설명 부분에서 이미지의 초기 grid를 S by S로 나눈다고 하였는데, 논문에서 S=7이 된다.  
	- 마지막 layer의 해상도로 7이 사용되므로, 그리드를 구성하는 과정에서 애초에 해상도의 사이즈로 결정하게 된다. 



YOLO는 당시 real time detection task에서 좋은 성능을 나타내었다. faster rcnn에 비해서 약 3배 가량의 속도를 보였지만, 정확도는 약간 낮은 모습을 보여주었다. 

![image](https://user-images.githubusercontent.com/38639633/110722974-a3182900-8256-11eb-82fc-03cf4cea3d39.png)





## Single Shot MultiBox Detector (SSD)

아무래도 yolo는 맨 마지막 layer에서만 한 번 prediction 하기 때문에, localization 정확도는 다소 떨어지는 모습이었다. 이를 보완하기 위해 구현된 방식이 `SSD`방식이다. 

SSD는 multi scale object를 더 잘 처리하기 위해 중간 feature map을 각 resolution에 적절한 bounding box를 출력할 수 있도록 `multi-scale 구조`를 만들었다. 

![image](https://user-images.githubusercontent.com/38639633/110723309-33566e00-8257-11eb-8b61-d979436116e7.png)

- 위 그림에서 볼 수 있듯,  8 x 8 feature map과 4 x 4 feature map에서의 바운딩 박스가 다름을 볼 수 있다. 
- 각 feature map마다 해상도에 적절한 바운딩 박스를 만들어 주는 것이 장점이다 



아키텍쳐를 살펴보면 다음과 같다. 

![image](https://user-images.githubusercontent.com/38639633/110723483-757faf80-8257-11eb-97fe-19741267e442.png)

- backbone인 vgg로부터 conv4블럭의 중간 feature map부터 마지막 출력 layer까지 classifier가 붙어있는 모습을 볼 수 있다. 

- 특징은 convolution이 진행되면서 발생하는 많은 conv layer에서 다양한 scale을 고려하여 classifier를 붙이는 모습이다. 

- 이전과 마찬가지로 클래스 수와 함께 4개의 coordinate를 출력해야하기 때문에, 각 위치마다 4개 혹은 6개의 (classes + 4)라고 되어 있는 것을 확인할 수 있다. 

- 또한 그림에서 마지막 부분, class마다 8732개의 디텍션이 이루어진다는 의미는 다음의 원리로 계산된다.

	![image](https://user-images.githubusercontent.com/38639633/110725154-6b12e500-825a-11eb-85a3-d0a7ad801909.png)

- 위와 같이 각 feature map마다 각 픽셀 위치에서 몇 개의 앵커박스가 존재하는지 그 수가 계산된다. 



![image](https://user-images.githubusercontent.com/38639633/110725307-ba591580-825a-11eb-92b0-f64606956e6d.png)

- SSD는 YOLO는 물론이고 Faster R-CNN보다 성능과 속도면에서 더 개선된 모습을 보여주었다. 



# Two-stage detector vs. one-stage detector





