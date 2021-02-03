# [DLBasic] Computer Vision Applications

Computer Vision 에서 CNN을 이용한 분야를 알아보고자 합니다. **Semantic segmentation**의 정의, 핵심 아이디어에 대해 배웁니다. **Object detection**의 정의, 핵심 아이디어, 추가적으로 종류에 대해 배웁니다. 



# 1. Semantic Segmentation

**문제정의** 

- 이미지의 모든 픽셀을 분류하는 문제

	![image-20210203171918154](https://user-images.githubusercontent.com/38639633/106746570-13cf9280-6666-11eb-9b18-b4ea1d254df0.png)

- Dense classification이라고도 불린다  

- 자율주행에 사용되는 핵심 기술이다. 

	![](https://www.jeremyjordan.me/content/images/2018/05/deeplabcityscape.gif)

	> refer : `https://www.jeremyjordan.me/content/images/2018/05/deeplabcityscape.gif`



<br>



## FCN(Fully Convolutional Network)

<br>



### difference Dense and Fully conv.

- Dense
	- 마지막에 tensor를 flatten 한 뒤에 dense layer를 통해 벡터를 일자로 펴준다. 
- Fully conv.
	- Convolutionalization이라고도 부른다.
	- flatten과 dense layer를 없애는 것 자체가 장점이다. 

<br>


### Convolutionalization

- Dense layer

	![image-20210203173834917](https://user-images.githubusercontent.com/38639633/106746576-1500bf80-6666-11eb-9292-da7102561ba9.png)

	> number of param. : 4 X 4 X 16 X 10 = `2,560`

- Fully conv.

	![image-20210203173944524](https://user-images.githubusercontent.com/38639633/106746577-1500bf80-6666-11eb-8765-d2ef7354f348.png){: .center}

	> number of param. : 4 X 4 X 16 X 10 = `2,560`

- 보이는 것과 같이 파라미터의 수에는 변화가 없다. 

<br>



### 특징

- Transforming fully connected layers into convolution layers enables a classification net to output a heat map.
- 즉, 인풋 이미지에 상관없이 네트워크가 돌아가는 것이 특징이다. 또한, output이 커지게 되면 뒷단의 네트워크가 따라서 커진다. 
	- 왜일 간격이 개같니

<br>



### FCN

- while FCN can run with inputs of any size, the output dimensions are typically reduced by subsampling.
- So we need a way to connect the coarse output to the dense pixels

<br>



### Deconvolution(conv transpose)



![image-20210203182624114](https://user-images.githubusercontent.com/38639633/106746579-15995600-6666-11eb-8fe0-c02663bd7e54.png){: width = "70%"}{: .center}

- 직관적으로는 conv 연산의 `역연산`이다.
- 하지만, 역연산이라는 건 실제로 불가능하다. 예를 들면, 어떠한 두 수를 더해서 10을 만들었다고 할 때, 10을 만드는 두 개의 수는 여러가지이다. 하지만, 거꾸로 10을 만드는 두 수를 찾으라고 한다면 이는 불가능하다.
	- convolution을 진행하면서 filter의 정보를 합해 더 작은 feature로 만들기 때문에, 이 방식의 역 연산은 불가능하다.
- 하지만 비슷한 size를 원하기 때문에, 많은 padding을 부여함으로써 이와 같이 deconvoluion 연산을 정의하게 된다. 



![](https://t1.daumcdn.net/cfile/tistory/99CA8E3359FE990510)

<br>



### Result

- 이와 같은 방식을 통해 sementic segmentation task를 진행한다. 
	<br>

---

<br>

# Detection

per pixel이 아닌 bounding box로써 이미지 중 어떠한 부분을 검출하는 분야이다. 

<br>

## R-CNN

![image-20210203184512116](https://user-images.githubusercontent.com/38639633/106746580-15995600-6666-11eb-881a-9aa4f85a541d.png)

R-CNN (1) tackes an input images, 92) extracts around 2,000 region proposals (using Selective search), (3) compute features for each proposal (using AlexNet), and then (4) classifies with linear SVMs.

<br>


## SPPNet

RCNN의 문제는 image의 bounding box를 뽑을 때, 그 모든 patch들을 CNN에 다 통과시켜야 한다는 점이다. 

![image-20210203185044785](https://user-images.githubusercontent.com/38639633/106746582-1631ec80-6666-11eb-912a-537e657a4364.png)

> In `R-CNN`, the number of crop / warp is usually over 2,000 meaning that CNN must run more than 2,000 times (59s/image on CPU).  
> However, in `SPPNet`, CNN runs once.

이미지의 patch를 추출(crop)하고, 이를 사이즈 조절(warp)한 뒤 conv-layer를 통과시키는 R-CNN은 각 patch를 **모두** conv에 통과시켜야 했지만, SPPNet의 경우 한 이미지만을 conv에 통과시킨 후, 각 부분별 **s**patial **p**yramid **p**ooling을 진행함으로써 속도를 개선시켰다. 


<br>

## Fast R-CNN

1. Take an input and a set of bounding boxes. (patch를 뽑는다 and region 정보 얻는다.)
2. Generated convolutional feature map(conv feature map 통과)
3. For each region, get a fixed length feature from ROI(Region of Interest) pooling(각각의 region에 대해서 fixed length feature를 뽑는다.)
4. Two outpus: class and bounding-box regressor(neural network를 통해 bounding box의 라벨을 찾는다. )

![image-20210203192439029](https://user-images.githubusercontent.com/38639633/106746585-1631ec80-6666-11eb-9d7c-6a9998d31da6.png)

![image-20210203195832312](https://user-images.githubusercontent.com/38639633/106746587-16ca8300-6666-11eb-8458-0ea68969c591.png)

> reference : cs231, 2017 lecture 1

CNN을 통과한 feature값에 image의 bounding box 부분에 대한 pooling을 진행하는 방식이다. 이후 이 pooling output을 FC-layer에 넣게 된다. 

<br>


## Faster R-CNN

Faster R-CNN = **Region Proposal Network** + Fast R-CNN

내가 bounding box를 뽑는 것도 학습을 통해서 뽑자!는 모델이다. 
<br>

### What is RPN?

이미지 안에서 지정된 바운딩 박스에 물체가 있는지 없는지를 찾아주는 알고리즘이다. 

![image-20210203201634172](https://user-images.githubusercontent.com/38639633/106746588-17631980-6666-11eb-90bc-222b6f83aa39.png){: .center}

여기서 `anchor box`는 미리 만들어 놓은 bounding box의 크기이다. 즉, 미리 템플릿을 만들어놓은 후 이 템플릿의 사이즈가 어떻게 바뀔지에 대한 학습을 진행한다. 

![image-20210203202010143](https://user-images.githubusercontent.com/38639633/106746592-17631980-6666-11eb-822a-c4b0de2a4bdf.png)

학습을 진행하면서 Fully conv.가 물체가 해당 바운딩 박스에 포함될지 여부에 대한 정보를 가지고 있게 된다. 

- RPN의 경우에는 `predefined region size가 9개` 있다. (128, 256, 512)의 사이즈를 가지는 region이 1:1, 1:2, 2:1)의 비율로 정한 뒤 9개의 region size 중 하나를 고르게 된다. 
- 각각의 바운딩 박스에 대한 정보는 `4개의 parameter`로 결정된다. (키우고 줄일지, width or height)
- 이 파라미터는 해당 바운딩 박스가 `쓸모 있는지 없는지` 여부를 판단한다. (whether to use it or not)

<br>


## YOLO v1

YOLO (v1) is an extremely fast object detection algorithm.

- baseline : 45fps / smaller version : 155fps

it **simultaneously** predicts multiple bounding boxes and class probabilities.

- No explicit bounding box sampling(compared with Faster R-CNN)

> 속도가 Faster R-CNN보다도 빠른 이유는 모델을 거친 뒤 나오는 것이 아니라, 딱 한장만을 보고 레이블링을 찍어내기 때문이다. You Only ~~Live~~Look Once!

![image-20210203203436584](https://user-images.githubusercontent.com/38639633/106746594-17fbb000-6666-11eb-8e81-0d68fd1cddf2.png)



1. Given an image, **YOLO** divides it into S$\times$S grid
	If the center of an object falls into the grid cell, that grid cell is responsible for detection.
2. Each cell predicts B bounding boxes(B=5). 
	Each bounding box predicts
	- box refinement (X / y / w / h)
	- confidence(of objectness)
3. Each cell predicts C class probabilities. (해당 바운딩 박스의 중점에 있는 object가 어떤 클래스인지를 예측한다. )
4. In total, it becomes a tensor with S$\times$S$\times$(B*5+C) size.
	- S$\times$S : Number of cells of the grid
	- B*5 : B bounding boxes with offsets  (X / y / w / h) and confidence
	- C : Number of classes