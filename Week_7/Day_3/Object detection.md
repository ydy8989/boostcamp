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

classification branch는 crossentropy loss를 사용하고, box coordinate regression문제는 regression loss를 사용하는데, 이 두 loss function이 RPN을 위한 것이고, 전체 target task를 위한 ROI별 classification에 대한 loss는 따로 또 하나가 추가되어 전체 구조가 End-to-End로 학습되게 되는 원리이다. 

