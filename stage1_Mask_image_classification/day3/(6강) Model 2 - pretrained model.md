# (6강) Model 2 - pretrained model

**강의 소개**

일반적으로, 모델을 새로 설계 하는 데는 시간이 걸립니다. 나는 당장 서비스를 해야 하는데, 모델 설계에 시간을 많이 쏟고 있다면 비효율적일 수 있습니다. 따라서, 기존에 검증된 우수한 모델 구조와 미리 학습된 weight를 재사용하는 방법에 대해서 다룹니다.

지금껏 Computer vision 분야에서 훌륭한 모델 아키텍쳐가 나올 수 있게 된 배경과, 그 모델들을 어떻게 우리 태스크로 가져와서 활용할 수 있을지, 활용할 때 주의사항은 없을지 등에 대해서 알아보도록 하겠습니다.



**Further Reading**

\- [Torchvisions Models](https://pytorch.org/vision/stable/models.html)



## Computer vision의 발전

계속적으로 컴퓨터 비전 분야의 발전으로 상당히 많은 일을 자동화 할 수 있다. 

![image](https://user-images.githubusercontent.com/38639633/113502230-e28c1980-9565-11eb-9020-d7befe94f0d2.png)



이러한 발전에 가장 많은 기여를 한 것은 바로 ImageNet이라고 할 수 있다. 

- 획기적인 알고리즘 개발과, 검증을 위해 높은 품질의 데이터 셋은 필수적이다. 
- About 14 million images
- About 20 thousands categories



**ImageNet이 만들어 지고 나서...**

![image](https://user-images.githubusercontent.com/38639633/113502260-2aab3c00-9566-11eb-94ab-2b80153c9c5f.png)

- 이미지 데이터 셋을 구축했기 때문에, 계속적으로 모델을 검증하고 비교할 수 있었으며 일반화시킬 수 있었다. 



## Pretrained Model

**모델 일반화를 위해 매번 수 많은 이미지를 학습시키는 것은 까다롭고 비효율적이다.**

![image](https://user-images.githubusercontent.com/38639633/113502322-a3aa9380-9566-11eb-809d-bcb981204af0.png)

- 성능 좋은 모델로 우리의 task를 고민해볼 수 있고, 미리 학습된 웨이트들을 통해 더 많은 데이터를 학습시킨 것과 같은 효과를 얻을 수 있다. 



**미리 학습된 좋은 성능이 검증되어 있는 모델을 사용하면 시간적으로 매우 효율적이다.**

![image](https://user-images.githubusercontent.com/38639633/113502373-fab06880-9566-11eb-9606-edcaf22282ce.png)

- 토치비전과 Timm에서 더 많은 pretrained model을 사용할 수 있다. 



**너무나도 손쉽게 모델 구조와 pretrained weight를 다운로드 할 수 있다. **

![image](https://user-images.githubusercontent.com/38639633/113502440-48c56c00-9567-11eb-9689-f8c18cb9019e.png)



## Transfer learning

학습된 weight를 가져온다고 하더라도 바로 내 task에 사용할 수는 없다. 나의 task에 맞게 수정하고 변형해줘야한다. 이처럼 이미지넷의 학습 웨이트를 '우리의 것'으로 가져오고 학습시키는 과정을 `Transfer learning`이라고 한다. 



### CNN base 모델 구조

**Input + CNN Backbone + Classifier $\Rightarrow$ Output**

![image](https://user-images.githubusercontent.com/38639633/113502527-c8ebd180-9567-11eb-8830-f81c39d34ba8.png)

### Code Check

**torchvision model 구조**

```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
print(resnet18)
```

![image](https://user-images.githubusercontent.com/38639633/113502547-ee78db00-9567-11eb-9173-a0111b3cd494.png)

- 우리의 경우 마지막을 1000개의 클래스로 나누는 task가 아니기에 이를 바꿔줄 필요가 있다. 



### 내 데이터, 모델과의 유사성

**ex) ImageNet Pretraining**

![image](https://user-images.githubusercontent.com/38639633/113502601-44e61980-9568-11eb-8b96-b94fab0cce62.png)

- 이미지 넷의 문제는 위와 같이 실생활에서의 object를 분류하는 것이다. 
- 하지만 우리의 데이터와 같을까?



**내가 설정한 문제와의 비교는 꼭 필요하다.**

![image](https://user-images.githubusercontent.com/38639633/113503082-98f1fd80-956a-11eb-8d51-5f1c039c3285.png)

- 내가 해야하는 task와 사용할 pretrained 모델의 데이터를 꼭 비교해보자. 



### Case by case

**case1. 문제를 해결하기 위한 학습 데이터가 충분하다.**

- 이 경우에는 pretrained 모델을 사용해도, 안해도 무방하다.

**case2. 데이터가 불충분하다.**

- 이 경우에는 미리 학습된 CNN backbone을 사용하는 것을 권장한다.





