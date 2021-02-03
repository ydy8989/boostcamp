# [DLBasic] Modern CNN - about 1x1 convolution

ILSVRC라는 Visual Recognition Challenge와 대회에서 수상을 했던 **5개 Network** 들의 주요 아이디어와 구조에 대해 배웁니다.

**Network list**

- AlexNet
	- 최초로 Deep Learning을 이용하여 ILSVRC에서 수상.
- VGGNet
	- 3x3 Convolution을 이용하여 Receptive field는 유지하면서 더 깊은 네트워크를 구성.
	- [Receptive field 참고 자료](https://cs231n.github.io/convolutional-networks/#conv)
- GoogLeNet
	- Inception blocks 을 제안.
- ResNet
	- Residual connection(Skip connection)이라는 구조를 제안.
	- h(x) = f(x) + x 의 구조
- DenseNet
	- Resnet과 비슷한 아이디어지만 Addition이 아닌 Concatenation을 적용한 CNN.

## 1.1. ILSVR

Imagenet Large-Scale Visual Recognition Challenge

- classification / detection / localization / segmentation

- 1000diff categories

- 1 million images

	| year  | Error Rate |
	| :---: | :--------: |
	| 2010  |    28%     |
	| 2011  |    25%     |
	| 2012  |    16%     |
	| 2013  |    11%     |
	| 2104  |     6%     |
	| 2015  |    3.5%    |
	| Human | about 5.1% |

## 1.2. AlexNet

**key ideas**

- ReLU activation
- 2 GPUs
- local response normalization, overlapping pooling
- data augmentation
- dropout

**relu activation**

- preserves properties of linear model
- easy to optimize with gradient descent
- good generalization
- overcome the vanishing gradient problem



## 1.3. VGGNet

![image-20210203213902867](http://localhost:4000/assets/img/boostcamp/image-20210203213902867.png)

**key ideas**

- 3 X 3 convolution. filter의 사이즈를 줄이기 시작했다.

- why?

	![image-20210203213624646](http://localhost:4000/assets/img/boostcamp/image-20210203213624646.png)

	- 3 by 3 filter를 두 번 사용하는 것은 5 by 5 filter를 한 번 사용하는 것과 I/O의 관점에서 별 차이가 없다.
	- 하지만 파라미터의 수를 계산해보면 약 10만개 가량이 차이난다.
	- 이는 계산속도와 generalization performance에 영향을 끼친다.



## 1.4. Googlenet

![image-20210203213931647](http://localhost:4000/assets/img/boostcamp/image-20210203213931647.png)

**inception blocks**

- what are the **benefits** of the `inception block`?

	- reduce the number of parameter

- How?

	- recall how the number of parameters is computed

	- 1 X 1 convolution can be seen as channel wise dimension reduction,

		> 즉 채널 방향으로 차원을 줄일 수 있다.

	![image-20210203214117290](http://localhost:4000/assets/img/boostcamp/image-20210203214117290.png)

	- 앞서 설명한 “작은 사이즈의 filter를 여러번 사용하는 방법”과 유사하게,
		layer는 깊게 만들고 I/O size는 유지하면서 파라미터의 사이즈는 줄일 수 있다.
	- 약 30% 가량으로 파라미터가 줄어든 모습을 확인할 수 있다.

**benefit of 1,1 conv**

- 1x1 convolution enables about 30% reduce of the number of pararmeters



## 1.5. comparing

**# of parameters?**

- `AlexNet(8-layers)` : 60M
- `VGGNet(19-layers)` : 110M
- `GoogLeNet(22-layers)` : 4M



## 1.6. ResNet

**deeper neural networks are hard to train**

- overfitting is usually caused by an excessive number of parameters
- bus not in this case.

**idea**

- add an identity map(skip connection)

	![image-20210203215048189](http://localhost:4000/assets/img/boostcamp/image-20210203215048189.png)![image-20210203215118424](http://localhost:4000/assets/img/boostcamp/image-20210203215118424.png)

- skip-connection을 사용하지 않을 때(**좌측**)는 더 깊은 신경망의 error가 더 큰 모습을 보여준다. 하지만, ResNet의 경우(**우측**), 신경망의 깊이가 깊어졌음에도 불구하고 얕은 신경망에 비해서 더 낮은 error율을 보여준다.
	이는 신경망이 깊어졌음에도 학습을 제대로 하고 있다는 것을 보여준다.

- add an identity map after nonlinear activations:

	- 차원을 맞춰주기 위해 1x1 conv를 이용한다.

- bottleneck architecture

	- ![image-20210203215606937](http://localhost:4000/assets/img/boostcamp/image-20210203215606937.png)

**History of Convolutional Network**

![image-20210203215740181](http://localhost:4000/assets/img/boostcamp/image-20210203215740181.png) 더 좋은 성능을 발휘하는 모델일 수록 파라미터의 수는 작아지는 방향으로 발전해왔다.



## 1.7. DenseNet

**DenseNet uses concatenation instead of addition**

![image-20210203220135858](http://localhost:4000/assets/img/boostcamp/image-20210203220135858.png)

- (좌측) ResNet의 경우, skip-connection을 수행할 때 `+` pairwise하게 더해준다.

- (우측) DenseNet의 경우, skip-connection을 수행할 때 `concat`을 통해 나열해준다.

- 이 때, 단순히 concat을 반복하게 되면 layer가 깊어질수록 아래와 같이 파라미터가 중첩되고, 마지막 층에 가서는 그 수가 기하급수적으로 늘어나게 된다.

	![image-20210203220541835](http://localhost:4000/assets/img/boostcamp/image-20210203220541835.png)

**Dense Block**

- Each layer concatenates the feature maps of all preceding layers
- the number of channels increases geometrically.

이를 해결하기 위해 아래와 같은 Transition block을 Dense block과 반복하며 파라미터를 조절해준다.

**Transition block**

- batchnorm -> 1x1conv -> 2x2 Avgpooling

- Dimension reduction

	![image-20210203221419645](http://localhost:4000/assets/img/boostcamp/image-20210203221419645.png)


​

## 1.8. Summary

**key takeaways**

> `VGG` : repeated 3××3 blocks
> `GoogLeNet` : 1××1 convolution
> `ResNet` : skip-connection
> `DenseNet` : concatenation