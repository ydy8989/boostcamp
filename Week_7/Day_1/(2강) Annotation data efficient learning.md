# (2강) Annotation data efficient learning

**강의 소개**

컴퓨터 비전 문제를 푸는 딥러닝 모델은 supervised learning으로 학습하는 것이 유리하다는 사실은 알려져 있습니다.
하지만, 딥러닝 모델을 학습할 수 있을 만큼 고품질의 데이터를 많이 확보하는 것은 보통 불가능하거나 그 비용이 매우 큽니다.

2강에서는 Data Augmentation, Knowledge Distillation, Transfer learning, Learning without Forgetting, Semi-supervised learning 및 Self-training 등 주어진 데이터셋의 분포를 실제 데이터 분포와 최대한 유사하게 만들거나,
이미 학습된 정보를 이용해 새 데이터셋에 대해 보다 잘 학습하거나,
label이 없는 데이터셋까지 이용해 학습하는 등 주어진 데이터셋을 최대한 효율적으로 이용해 딥러닝 모델을 학습하는 방법을 소개합니다.

**Further Reading**

\- CutMix : [https://arxiv.org/pdf/1409.1556.pdf](https://arxiv.org/pdf/1905.04899.pdf)



## Data augmentation

### learning representation of dataset

- NN는 컴퓨터가 이해할 수 있는 형태로 압축한 모델이다

- 데이터를 통해 패턴을 분석하는 것이기에 편향될 수 밖에 없다(통계적으로 데이터가 많은 쪽으로 예측이 기운다.)

- 하지만, 실제 데이터는 bias되어있다.

	![image](https://user-images.githubusercontent.com/38639633/110277204-239d1680-8018-11eb-98bb-0e609d7e1b53.png)

- 위 이미지들은 어느 데이터 셋에 들어있는 데이터들의 평균 이미지를 표현한 이미지이다. 

	- 순서대로 **클래스별 평균, 보스턴 찰스강 영상들의 평균, 공원에 관련된 이미지들의 평균**
	- 공원의 이미지를 보면 사람이 가운데에 있는데, 일반적으로 사람을 가운데에 피사체로 두고 찍었다는 의미이다.



The training dataset is sparse samples of real data

- The training dataset contains only fractional part of real data

	![image](https://user-images.githubusercontent.com/38639633/110277591-e5542700-8018-11eb-90bc-1b53b7eaabb9.png)

- 우리가 보는 데이터는 실제로 빈공간이 많아 test 데이터에 대한 확신이 어렵다. 

- 더 큰 문제는 심지어 bias까지 되어있는 점이다. 



**The training dataset and real data always have a gap**

- Suppose a training dataset has only bright images

- During test time, if a dark images is fed as input, the trained model may be confused

	- problem : 데이터 셋은 fully하게 real data를 커버하지 못한다. 

	![image](https://user-images.githubusercontent.com/38639633/110277780-51368f80-8019-11eb-9653-24dfb53871c4.png)



**Augmenting data to fill more space and to close the gap**

- 결국 학습 단계에서의 sample을 조금이라도 변형을 더 줘서 빈 공간을 매꾸는 작업이 `Data augmentation`이다. 

	![image](https://user-images.githubusercontent.com/38639633/110277929-9a86df00-8019-11eb-9d63-760295fa139a.png)

### Data augmentation

**다양한 방법의 augmentation**

- applying various image transformations to the dataset
	- crop, shear, brightness, persepective, rotate....
- opencv and numpy have various methods useful for data augmentation
- 목표 : 학습 데이터의 분포를 real 데이터의 분포화 비슷하게!!



### Various data augmentation methods

**brightness adjustment**

![image](https://user-images.githubusercontent.com/38639633/110279419-77a9fa00-801c-11eb-84db-c135f15a7ca2.png)

**Rotate,flip**

![image](https://user-images.githubusercontent.com/38639633/110279482-855f7f80-801c-11eb-8b4e-fc584a28c054.png)

![image](https://user-images.githubusercontent.com/38639633/110279514-90b2ab00-801c-11eb-8197-fcdd66e8f0eb.png)

**crop**

![image](https://user-images.githubusercontent.com/38639633/110279550-9a3c1300-801c-11eb-909c-4e5c834ed1c7.png)

![image](https://user-images.githubusercontent.com/38639633/110279583-a32ce480-801c-11eb-8b2f-c21768dfead9.png)



**Affine transformation**(shear transform)

- preserves `line`, `length ration`, and `parallelism` in image

	- 즉, 가로 세로 비율이 일정하게 유지한 상태로 변화를 준다고 생각하면 이해가 쉽다.

- for example, transforming a rectangle into a parallelogram(See the shear transform example below)

	![image](https://user-images.githubusercontent.com/38639633/110279669-d3748300-801c-11eb-8038-a019c76c99eb.png)

	![image](https://user-images.githubusercontent.com/38639633/110279682-d8d1cd80-801c-11eb-88bc-b5ff60776dc1.png)

	![image](https://user-images.githubusercontent.com/38639633/110279697-e12a0880-801c-11eb-9f58-829c6d2f0a0a.png)

- 위와 같은 이동에서 `warp`이라는 워딩을 기억해놓자



### Modelrn augmentation techniques

**CutMix**

- `cut` and `mix` training example to help model bettor localize objects

	![image](https://user-images.githubusercontent.com/38639633/110279787-09b20280-801d-11eb-9acb-6caf9df063c6.png)



**generating new training image**

- mixing both images and labels(비율적으로 섞는..)

	![image](https://user-images.githubusercontent.com/38639633/110279838-1e8e9600-801d-11eb-84fe-8f47a709db08.png)

	

**RandAugment**

> 여러 aug방법을 조합하고 자동으로 searching하여 진행하는 방법이다. 

- Many augmentation methods exist. Hard to find best augmentations to apply

- **Automatically finding the best sequence of augmentations to apply**

- Random sample, apply, and evaluate augmentations

	![image](https://user-images.githubusercontent.com/38639633/110281823-c35ea280-8020-11eb-91d9-462a6f80488e.png)

- Example of augmented images in RandAug

	- Augmentation policy has two parameters(아래 두개가 파라미터임.)

		- which augmentation to apply(어떤 aug방법을 적용할 것인지)

		- magnitude of augmentation to apply(how much to augment)(얼마나 세게 적용할것인지?)

			![image](https://user-images.githubusercontent.com/38639633/110282063-2bad8400-8021-11eb-842e-8aeb9acfe11a.png)

	- parameters used in the above example

		- which augmentation to apply : shearX & autoContrast
		- Magnitude of augmentation to apply : 9

- Randomly testing augmentation policies

	- Finding the best augmentation policy(이러한 하나의 어그멘테이션 기법을 `policy`라고 부른다)
		- sample a policy : policy = {N augmentations to apply} by random sampling
		- 해당 policy로 학습및 evaluation을 진행한다. 

- augmentation helps model learning

	- 손쉽게 성능을 올릴 수 있음을 볼 수 있다. 

		![image](https://user-images.githubusercontent.com/38639633/110282343-ae364380-8021-11eb-8880-e39da1f6a6c6.png)



## Leveraging pre-trained information

### Transfer learning

- 좋은 퀄리티의 데이터를 얻는 것은 비용적으로도 비싸고, 획득하기에도 어렵다.
	- 교사학습은 학습에 많은 양의 데이터 셋이 요구된다. 
	- 또한, annotating은 매우 정교해야하며, 비싸다. 
- 이러한 문제를 해결하기 위한 좋은 방법은 `전이학습(transfer learning)`이다. 
	- A practical training method with a small dataset!



**Benefits when using transfer learning**

- 전이학습을 통해 우리는 연관된 새로운 task에 적은 노력으로도 높은 성능을 발휘할 수 있다.



**Motivational observation: Similar datasets share common information**

- transfer learning이 가능한 이유는 뭘까?

	- 아래 데이터는 서로 다른 데이터셋이다. 
	- 하지만, 각 데이터셋에는 '잔디'나 '자동차 바퀴'와 같은 공통적 요소가 포함되어있다. 
	- 이러한 것들이 이 전이학습을 가능케한다.

	![image](https://user-images.githubusercontent.com/38639633/110283909-34ec2000-8024-11eb-8e41-6c334b6c6eb4.png)



**2가지 방식의 transfer leanring**

1. Transfer knowledge from a pre-trained task to a new task

	- 10개 클래스에 대한 pre-trained 모델이 주어졌을 때,
	- pre-trained 모델의 마지막 레이어를 잘라내고, 새로운 task에 대하여 추가적으로 layer를 붙인다. 

	![image](https://user-images.githubusercontent.com/38639633/110284175-b774df80-8024-11eb-8a43-42f8fc36fc46.png)

2. Fine-tuning the whole model

	- pre-trained 모델이 데이터셋에 대하여 주어지면

	- 바꿀 마지막 layer에 대해서만 새로운 모델을 위한 layer로 교체

		![image](https://user-images.githubusercontent.com/38639633/110284383-14709580-8025-11eb-9b78-69ac8f8ea694.png)

	- conv layer에서는 정확히 학습이 되도록 learning rate를 낮게 설정하고, 새로 바꾼 task에서는 빨리 적응하도록 fully connected layer에서는 high learning rate로 학습하는 방식이다. 

	

	

### Knowledge distillation

**Passing what model learned to ‘another’ smaller model (Teacher student learning)**

- teacher network의 지식을 작은 사이즈의 student network에 주입하여 학습하는 방식이다. 
- `Distillate(증류)`는 모델 압축에 유용하게 쓰이는 방법이라고 할 수 있다. (큰 모델이 아는 것을 작은 모델이 모방 $\rightarrow$ 이말은 압축과 일맥상통) 
- **Also, used for pseudo labeling (Generating pseudo labels for an unlabeled dataset)**
	- 즉, teacher network에서 추출한 output을 가짜 레이블로 자동생성하는 메카니즘으로 사용하기도한다.
	- 가짜 레이블로 자동 생성한 뒤에 더 큰 student를 사용할 때에도 regularization의 역할로 사용되게 하기도 한다. 
		- [https://analyticsindiamag.com/pseudo-labelling-a-guide-to-semi-supervised-learning/](https://analyticsindiamag.com/pseudo-labelling-a-guide-to-semi-supervised-learning/)여기를 참조해보자.



**Teacher-student network structure**

- student network는 teacher network가 알고 있는 것을 배운다. 

- student network는 teacher network의 output을 모방한다.

- 아래 그림을 살펴보자

	![image](https://user-images.githubusercontent.com/38639633/110317192-21a27a00-804f-11eb-9547-5fdd46d31d43.png)

	- pre-trained 네트워크를 teacher model로써 준비한다.
	- 아직 학습이 되지 않은 student model을 initialization한다. 
	- 이 때, student는 더 작은 모델을 쓰는 것이 일반적이다
	- 같은 입력 input_X에 대해서 동시에 output으로 뽑고
	- 이 두 output을 KL divergence loss로 계산하고, backprop을 진행한다.
	- 여기서 backprop은 아래쪽 student 모델로만 진행하며 학습시킨다. 

- 위 과정을 진행하며 label은 전혀 사용하지 않았기에 unsupervised learning이라고 할 수 있다. 



**Knowledge distillation**

만약 label을 가진 데이터도 존재할 때는 어떻게 할까?

![image](https://user-images.githubusercontent.com/38639633/110317840-05eba380-8050-11eb-83f3-3684ec6a5f75.png)

- 위 그림에서 `Distillation loss`는 teacher model과 이를 따라하며 학습된 student model의 차이를 의미한다. 
- `Student loss`는 레이블이 있기에 존재하는 ground truth와 student model의 output의 차이를 의미한다. 



**Hard label vs Soft label**

Student model의 output인 `soft prediction`은 무엇을 의미하는 것인가? 

- 일반적인 원-핫-벡터가 아닌, softmax의 output으로 나오는 값을 soft prediction이라고 말한다. 
- 즉, 클래스별 confidence를 더하여 1이 되는 예측값을 말한다.



**softmax with `temperature`(T)**

![image](https://user-images.githubusercontent.com/38639633/110319436-3c2a2280-8052-11eb-9692-0659f7fca6a4.png){:width="50%"}
![image](https://user-images.githubusercontent.com/38639633/110319451-43e9c700-8052-11eb-97a5-237b4d2e0e08.png){:width="50%"}

- 위와 같이 T로 나눠줌으로써 softmax를 smoothing 시켜주는 효과를 얻을 수 있다. 
- softmax의 결과처럼 1에 근접한 값이 한번 나오면 그 이후의 값들은 모두 아래 깔린다. 
- 이는 정보를 협소하게 얻는 원인이 된다. 
- 반면에 T로 나눔으로써 값을 smoothing 시킬 경우, 정보의 상대적 비교는 가능하되 그 총량을 넓게 바라볼 수 있게 도와준다. 



특히, knowledge distillation task에서 **Semantic information은 고려하지 않는다**

- teacher의 output의 각각의 dimension이 pre-trained에 사용된 이전의 task들과 연관이 되어있긴하다. 
- 하지만, student loss로 학습되는 데이터 셋은 pre-trained 모델에 사용되는 데이터 셋과 전혀 다른 task일 수도 있다. 
	- 예를들면 레이블을 공유하지 않는다던가
	- 카테고리가 겹치지 않는다던가..
- 그래서 중복되는 정보가 없더라도 
- soft label에 사용되는 각각의 의미가 중요하다기 보다는 
- 전체의 개형이 추상적 지식을 표현하고 있어서, student model의 output인 soft prediction이 그것(추상적 정보를 내포하는 개형 혹은 분포 그 자체)을 따라하는 것이 중요하다는 것이다
- 각각 element의 의미가 distillation 과정에 많은 영향을 주는 것은 아니라는 점이다. 



**intuition about distillation loss and student loss**

둘다 classification의 경우 cross entropy 계열의 loss를 사용하긴 한다. 

 ![image](https://user-images.githubusercontent.com/38639633/110322197-156deb00-8056-11eb-92b3-8f47a8c146a4.png)



최종적으로 label이 있는 데이터를 사용할 때는 `Distillation loss`와 `Student loss`를 합쳐서 사용하고, backpropagation은 두 개의 loss로부터 student model 방향으로 진행되어 해당 모델만을 학습하게 된다. 



## Leveraging unlabeled dataset for training

### Semi-supervised learning

- 이 방식은 unlabeled data를 목적성 있게 잘 사용하기 위함이다. 
- 왜 언레이블드 데이터일까? 
	- supervised data는 label을 구축하는데 한계가 있다. 
- 준지도학습은 말 그대로 약간의 레이블 데이터를 활용하여 unlabeled data를 활용할 수 있는 방식이다. 

---

![image](https://user-images.githubusercontent.com/38639633/110322668-d3917480-8056-11eb-9f64-4b00b4dacae8.png)

- 적은 양의 labeled data를 학습하여 model을 만든다
- 해당 모델을 통해 unlabeled data의 Pseudo label을 잔뜩 생성한다. 
- 즉, 엉터리 label인 pseudo-labeled data와 진짜 labeled data를 합쳐서 재학습을 진행한다. 

위와 같은 방식과 더불어 앞서 소개한 data augmentation skill과 knowledge distillation 등을 결합해 새로운 지평을 연 모델을 이어서 소개하도록 하겠다.



### Self-training

이 방식은 [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/pdf/1911.04252.pdf)에서 처음 등장한 방식이다. 

- Augmentation + Teacher-student networks(knowledge distillation) + semi-supervised learning 세가지 방식이 결합된 모델이며

- 2019년 imagenet 분류 태스크에서 SOTA를 달성했다.

	![image](https://user-images.githubusercontent.com/38639633/110324537-75b25c00-8059-11eb-9b41-291ba7a83e5f.png)

학습 방식 및 architecture를 간략히 도식화한 다이어그램을 살펴보자

![image](https://user-images.githubusercontent.com/38639633/110324710-b01bf900-8059-11eb-9f65-c7b609830971.png)

간략히 설명하자면 다음과 같다.

- 1M개의 Labeled data와 300M개의 Unlabeled data를 준비한다.
- 1M의 Labeled data를 학습하여 만든 model1으로 300M의 unlabeled data에 대한 pseudo-labelling을 실시한다. 
- 이렇게 300M의 데이터에 레이블이 생기면, 1M의 labeled data와 합쳐 randAugment를 진행한다. 
- 301M보다 많아진 RandAugment output으로 새 모델 Model2를 학습한다. 
- 이 모델에 다시 300M의 unlabeled data로 pseudo-labelling을 실시한다. 
- 반복한다...

