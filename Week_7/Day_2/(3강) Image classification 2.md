# (3강) Image classification 2

**강의 소개**

이번 강의에서는 1강 Image Classification에 이어서 대표적인 CNN 모델들에 대해 배웁니다.

먼저 VGGNet과 비슷한 시기에 등장한 GoogLeNet을 시작으로, 지금도 많이 쓰이고 있는 ResNet에 공부하고 실습을 진행합니다. 이 외에도 추가적으로 몇가지 CNN 모델들에 대한 소개를 합니다. 특히, 1강과 3강까지 다룬 4가지 모델 (AlexNet, VGGNet, GoogLeNet, ResNet)에 대하여 메모리 측면과 계산 효율 관점에서 비교 분석을 합니다.

**Further Reading**

\- ResNet: https://arxiv.org/pdf/1512.03385.pdf

 

## CNN architectures for image classification

강의에서는 다양한 아키텍쳐를 다뤘다. AlexNet, VGG, GoogleNet을 다뤘는데, 이 세 모델에 대한 내용은 따로 언급하지 않겠다. 그 이유는 앞서 정리했던 [포스팅](https://ydy8989.github.io/2021-02-03-cnn/)에서 어느정도 언급을 했기 때문에 새로이 자세하게 다루는 `ResNet`과 그 이후 모델에 대한 내용만을 다룰 예정이다. 



### ResNet

![image](https://user-images.githubusercontent.com/38639633/110426732-f02abc80-80e9-11eb-8045-fb432ec79558.png)

- resnet은 2016 CVPR에 등장한 논문으로 **residual connection**을 처음 등장시킨 논문이다
- 최근까지도 Backbone과 실험을 resnet으로 먼저 할만큼 주요한 논문중 하나이다. 

#### revolutions of depth

- resnet 논문의 주요 연구 성과는 깊은 층쌓기이다. 
- 기존에도 연구자들이 많은 노력을 했지만, 층을 깊게 쌓지는 못했다. 
- 그 이유는 무엇인지, resnet의 성과가 무엇인지 살펴보자. 



#### Degradation problem

- as the network depth increases, accuracy gets saturated$\Rightarrow$rapidly

![image](https://user-images.githubusercontent.com/38639633/110456090-82dc5300-810c-11eb-8d2a-ba01a2c573e0.png)

- 논문에서는 training error와  test error를 비교하였다.
- 이전까지는 모델 파라미터가 많을 수록 training에러가 더 낮아질 것으로 생각해왔었다. 
- 여기서 중요한점은 레이어 56 모델이 레이어20 모델보다 error가 높다는 것이 `아니다`
- train과 test 모두 일정 에러 아래로 안내려 가고 어느정도 선에서 수렴하는 모습을 보이는데 이것도 중요한 점이 아니다. 
- test에서 20층보다 56층짜리가 error가 더 높은데, 이 이유가 overfitting 때문이었다면 train에서 56layer 모델이 20layer 모델보다 더 작았어야 했다. 
- 그래서 적어도 학습 데이터에만 국한된 좋은 성능을 보여야만 했는데, 학습데이터에 대해서 56layer가 20layer보다 크다는 것은 아직 학습이 덜 되었다는 것을 의미한다. 
	- 혹은, 학습이 더될 여력이 있지만, 모델의 한계로 단지 수렴만 시키는 정도로 유지하는 것이다. 
- 이 결과가 overfitting에 대하여 counterintuitive한 관찰이었다. 
- 본 논문에서 저자들은 이러한 현상의 원인을 overfitting이 아닌 `degradation`이라는 다른 문제이고, 최적화 이슈로 인해 학습이 잘 안되었다고 결론지었다.



#### Hypothesis

- plain layer : input $x$에서 $H(x)$로 다이렉트로 학습하는 것은 매우 어렵다고 판단.
- residual block을 도입하여 $x$가 $H(x)$가 되는데 변화하는 정도($H(x)-x$)만큼만을 학습하도록 설계하는 것이 더 학습에 도움이 될 것이라고 가설을 설정하였다. 

![image](https://user-images.githubusercontent.com/38639633/110472482-c5a82600-8120-11eb-9eb6-95b979e80500.png)

- 이러한 방식을 구현하기 위해서 오른쪽 그림에서처럼 `shortcut connection(skip connection)`을 구현하였다. 

	> - Use layers to fit a residual mapping instead of directly fitting a desired underlying mapping  
	> - The vanishing gradient problem is solved by shortcut connection
	> - Don't just stack layers up, instead use shortcut connection  

- `역전파`관점에서도 우측 그림에서 $F(x)+x$에서 $x$ 방향으로 역전파가 계산될 때 두 가지 방향으로 흐를 수 있도록 설계를 하였다. 

	- weight layer를 통과할 때 vanishing gradient 현상이 일어나도 identity 방향에서는 살아있기 때문에, 학습 가능한 chance를 얻을 수 있다.



#### Analysis of residual connection

- 이 같은 residual block이 왜 성능이 좋을까?

- 한 분석 논문에 따르면 $2^n$ 의 경우의 수로 gradient가 지나갈 수 있는 방법이 생기기 때문이라고 한다. 

	> During training, gradients are mainly from relatively shorter paths  
	> Residual networks have $\mathcal{O}(2^n)$ implicit paths connecting input and output, and adding a block doubles the number of paths.

	- block 하나당 두 가지 방향이 생기기 때문에..



#### PyTorch code for ResNet

![image](https://user-images.githubusercontent.com/38639633/110476068-ee321f00-8124-11eb-8b4b-f7cab84d90be.png)

- resnet18을 기준으로 basicblock들을 생성하고, layer의 갯수는 두배씩 [2,2,2,2]로 준다. 

![image](https://user-images.githubusercontent.com/38639633/110476176-0bff8400-8125-11eb-8b11-43860058db22.png)

- layer는 위와같이 쌓이고, 각각의 layer는 아래와 같이 정의된다. 

![image](https://user-images.githubusercontent.com/38639633/110476183-0dc94780-8125-11eb-960f-f2f1d6e5046c.png)

- `_make_layer`라는 함수를 정의하여 파라미터에 따라 쌓이는 규칙을 정의할 수 있게하고, stride는 계속 2로 두어 공간 해상도를 절반씩 줄여간다. 반면에 채널의 사이즈는 두배로 늘어난다. 

![image](https://user-images.githubusercontent.com/38639633/110476190-0f930b00-8125-11eb-8da5-2b6750d06248.png)

- for loop을 통해 블럭을 sequential하게 쌓는다. 
- 이렇게 구성해두면 쉽게 관리할 수 있다.

![image](https://user-images.githubusercontent.com/38639633/110476237-17eb4600-8125-11eb-98b5-357a476ab56a.png)

- 마지막에는 linear layer로 마무리한다. 



### Beyond ResNets

Resnet 이후에도 다양한 시도들이 있어왔다. 이에 대한 case study를 간략이 하고 넘어가자.

#### DenseNet

- ResNet에서는 skip connection을 통한 indentity mapping을 추가했다면, 

- DenseNet에서는 `Channel 축으로 concatenate`를 하도록 설계 되어있다.

	> In the `Dense blocks`, every output of each layer is concatenated along the channel axis.
	>
	> - Alleviate vanishing gradientproblem
	> - Strengthen feature propagation
	> - Encourage the reuse of features

- 이전의 모든 connection을 이어주는 방식으로 **dense**하게 설계되어있다. 

![image](https://user-images.githubusercontent.com/38639633/110477565-aad8b000-8126-11eb-9392-a9700d2641b8.png){:width="60%"}

- 상위 레이어에서도 하위 레이어의 특징을 재참조할 수 있게된다. 
- 주의해야할 점은 resnet이 `+`였다면 Densenet은 `concatenate`라는 점이다. 
	- `+`는 신호를 합치는역할
	- `concatenate`는 옆으로 단순히 이어붙임으로써 신호는 보존한다. (단, 채널은 늘어난다.)



#### SENet

- depth를 높이거나 커넥션을 새로하는 방법이 아니라 현재 주어진 activation 간의 관계가 명확해지도록 채널간 관계를 모델링하고 중요도를 파악해서 특징을 attention할 수 있게끔하는 방식이다. 
- recalibrate channel-wise responses by modeling interdependencies between channels
- Squeeze and excitation operations(attention을 생성하는 방식은 두 가지가 있다.`             )
	- squeeze : captureing distributions of channel-wise responses by global average pooling
		- global average pooling을 통해 공간정보를 없애고 분포를 구한다(magnitude)
		- h와 w를 없애는 방향으로 channel의 평균 정보만을 취합하여 vertor로 만든다. 
	- Excitation : gating channels by channel-wise attention weights obtained by a FC layer
		- 이후 채널간의 연관성을 고려해서 W를 계산하고 attentioning을 한다. 
- 이후 다시 rescaling하여 텐서를 재가공한다.(색깔별로 중요도를 의미한다.)

![image](https://user-images.githubusercontent.com/38639633/110478734-fdff3280-8127-11eb-93c1-cadb5afa531c.png)



#### EfficientNet

2019년에 제안된 방식으로, 이전까지는 다음 세 가지 방식 중 하나로 모델의 성능을 개선시켜왔다. 

![image](https://user-images.githubusercontent.com/38639633/110484440-ff335e00-812d-11eb-9b2e-040490726ae5.png)



