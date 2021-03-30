# (3강) Dataset

**강의 소개**

데이터 전처리의 개념에 대해서 설명합니다. 컴퍼티션의 경우 어느 정도 정제된 데이터가 주어지지만, 앞으로 우리가 만나게 될 데이터는 cleaning 되어 있지 않는 경우가 많기 때문에 사전에 여러가지 작업들이 필요합니다. 이미지 데이터는 비교적 전처리할 거리가 그렇게 많지는 않지만 정형데이터나 텍스트 데이터의 경우는 상상을 초월하는 전처리를 경험하시게 될거에요...

그리고, Generalization(일반화) 관점에서 생각해 볼 수 있는 몇 가지 Skill를 다룹니다.

이러한 과정에서 의사결정을 하는데에는 앞서 많이 말씀 드린 것 처럼 문제를 어떻게 정의했느냐가 매우 중요한 요소로 활용될 수 있습니다.

 

**Further Reading**

\- [Albumentation: Augmentation 오픈소스 프로젝트](https://github.com/albumentations-team/albumentations.git)

 

## overview

ML에서의 `DATASET`이란 주어진 vanilla data를 모델이 좋아하는 형태로 만드는, 만든 것을 의미한다. 

![image](https://user-images.githubusercontent.com/38639633/112918094-0ce16f80-913f-11eb-99b8-e2727bf3d74c.png)

결과적으로 우리가 저장한 형태의 데이터를 아래와 같이 변경함에 그 목적이 있다. 

![image](https://user-images.githubusercontent.com/38639633/112918445-da844200-913f-11eb-88e2-94c0a9f526c4.png)



## Preprocessing

![image](https://user-images.githubusercontent.com/38639633/112918610-38188e80-9140-11eb-966e-103bdfc656f5.png)

- 가장 많은 비중을 차지하는 작업이 전처리 과정이다. 
- 실제 현업에서는 결측치, 이상치 등 많은 난항을 겪을 수 있다. 
- 컴퍼티션 데이터의 경우 잘 정제되어 있어 많은 작업이 필요하진 않다. 



**Bounding box**

"가끔 필요 이상으로 많은 정보를 가지고 있기도 한다..."

![image](https://user-images.githubusercontent.com/38639633/112918867-dad10d00-9140-11eb-9a01-e833fb8cce0a.png)

- 바운딩 박스를 통해 원하는 타겟을 데이터의 품질을 높이고, 이외의 배경을 노이즈로... 



**Resize**

"계산의 효율을 위해 적당한 크기로 사이즈 변경"

원본의 사이즈를 조절함으로써 효율적으로 학습하게 유도하고, GPU 리소스의 한계를 최소화한다. 



**Example : APTOS Blindness Detection**

"도메인, 데이터 형식에 따라 정말 다양한 전처리 case가 존재한다."

![image](https://user-images.githubusercontent.com/38639633/112919176-85e1c680-9141-11eb-8ba8-8e57a981683f.png)

- CT-images의 segmentation task에서 hounsfield unit 조정처럼 명도 채도를 변경하고, 디텍팅이 더 잘되게 하는 방식의 전처리가 존재한다.
- 각 도메인마다 독특한 방식의 전처리 방식이 존재하며 특히, medical 분야에서 많이 존재한다. 



## Generalization

**Bias and Variance**

"학습이 너무 안됐거나, 학습이 너무 잘됐거나..."

![image](https://user-images.githubusercontent.com/38639633/112919896-e3c2de00-9142-11eb-9794-3335cb926ee8.png)



**Train / Validation**

훈련 셋 중 일정 부분을 따로 분리, 검증 셋으로 활용한다. 

![image](https://user-images.githubusercontent.com/38639633/112921153-40bf9380-9145-11eb-82ef-8d64e53d3b71.png)

- 데이터가 작아지므로 모델의 성능을 안좋게 한다고 생각할 수 있지만, 그렇지 않다.
- `일반화`의 관점에서 꼭 검증 셋을 분류해야만 한다. 



**Data Augmentation**

주어진 데이터가 가질 수 있는 Case와 state의 다양성..

 ![image](https://user-images.githubusercontent.com/38639633/112921604-073b5800-9146-11eb-80bd-726ebd2c2797.png)

- test 셋에서 나올법한 가능성이 높은 데이터를 생성하고, 그 분포를 학습한다면 더 좋은 모델을 만들 수 있지 않을까?라는 물음에서 시작하였다.
- 문제가 만들어진 배경과 모델의 쓰임새를 살펴보면 힌트를 얻을 수 있다. 



**torchvision.transforms**

다양한 방식의 augmentation을 할 수 있는 파이토치의 함수

![image](https://user-images.githubusercontent.com/38639633/112922047-c728a500-9146-11eb-8aa0-737247a7ee05.png)

- 각각의 augmentation의 분포가 학습에 필요한지를 명확히 하고 넘어가자
	- 마스크 데이터의 경우 상하flipping은 사실상 의미가 없다.
	- 데이터가 거의 정자세로 찍혀 나오기 때문에...

![image](https://user-images.githubusercontent.com/38639633/112922229-166ed580-9147-11eb-95b7-a74dec5ef842.png)

- 다양한 방식의 augmentation이 가능하고, 손쉽게 Compose로 묶어 사용할 수 있다. 



**Albumentations**

transforms보다 더 빠르고 더 다양한 augmentation 방식을 제공한다. 

![image](https://user-images.githubusercontent.com/38639633/112922921-479bd580-9148-11eb-876b-cef284e4235b.png)

> [https://github.com/albumentations-team/albumentations#benchmarking-results](https://github.com/albumentations-team/albumentations#benchmarking-results)

