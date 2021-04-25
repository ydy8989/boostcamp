# (5강) Model 1

**강의 소개**

이제 본격적으로 모델을 디자인하는 과정입니다. 앞서 U stage에서 많이 다뤄보셨겠지만,

이 강의에서는 파이토치라는 프레임워크가 어떤 특징을 가지고 있는지와,

파이토치가 어떻게 모델 그래프(체인)을 구성하게 되고, 파라미터 값을 어떻게 저장하고 있는지, forward라는 함수의 동작 과정에 대해서 약간 디테일 하게 접근합니다.



**Further Reading**

\- [파이토치 Module 문서](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)



## model

What is model?

- "In general, a model is an infromative representation of an object, person or system"



## Design model with pytorch

### Pytorch

![image](https://user-images.githubusercontent.com/38639633/113473954-bdd16c80-94a7-11eb-8722-07c07db0ce2a.png)

- 파이토치의 장점은 1. 파이토닉, 2. 연구하기 좋고, 3. 다루기 쉽다. 
- 슬로건을 보면 알 수 있듯이, 연구(research)에 특화되어 있다. 



**Low-level, Pythonic, Flexibility**

![image](https://user-images.githubusercontent.com/38639633/113475054-5c60cc00-94ae-11eb-9b65-8f078375cfbf.png)

- 좌측 리모컨은 사용자 친화적이다. 반면에 우측 빵판은 어떤 의미인지 쉽게 이해하기 어렵다.
- 사용성의 측면에서는 좌측이 좋지만, 실험적 측면에서는 우측이 더 좋다. 
	- 파이토치는 우측과 비슷하게 'low level'로 구성되어있는 프레임워크이다. 
	- 재사용성이 좋고, 수정하거나 변형하기 용이하다. 

- Keras는 좌측과 같이 사용성은 좋지만, 변형이 어렵다.

![image](https://user-images.githubusercontent.com/38639633/113475295-ae562180-94af-11eb-9fce-774df3a54c9b.png)

- (좌측) 케라스 (우측) 파이토치
- 케라스는 사용하기는 쉽지만, 뜯어고치기는 힘들어서 파이토치의 이런점은 학습에 매우 도움된다. 



### nn.Module

**pytorch 모델의 모든 레이어는 nn.Module 클래스를 상속받아 따른다.**

```python
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.module):
    def __init__(self):
        super(MyModel, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))        
```

- `__init__`에서 클래스의 레이어들을 정의하고, `forward(self, x)`에서 레이어의 층을 쌓기!





![image](https://user-images.githubusercontent.com/38639633/113476577-46a3d480-94b7-11eb-8738-1b01c89a7cd4.png)

- 모델의 클래스를 a로 선언하고 `.modules()`라는 method를 통해 모듈들이 어떻게 구성되어 있는지 확인할 수 있다. 
- 여기서의 `.modules`는 모델의 weight를 머금고 있는. 즉, 파라미터를 가진 모듈을 의미한다. 
- 이 말은 `.modules`를 통해 파라미터를 저장하고 있는 저장소를 컨트롤할 수 있다는 의미이다. 
	- 이를 통해 모델 커스텀이 가능하다. 



### forward

**위의 MyModel (모듈)이 호출 되었을 때 실행되는 결과**

```python
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.module):
    def __init__(self):
        super(MyModel, self).__init__()
		self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x)) 
```

```python
x = torch.rand((1,3,256,256))
y = a(x)
y.shape
>>> torch.Size([1, 20, 248, 248])
```

- 모델 클래스의 `__init__`에서 정의되는 객체들이 모델이 호출된 후 input에서 output으로 가기까지 만들어지는 과정을 `forward`에서 정의하게 된다.
- 여기서 `a = MyModel()`로 정의했을 때, `a(x)`를 하는 것과 `a.forward(x)`를 실행하는 것의 결과는 동일하다. 



### nn.Module Family

**nn.Module을 상속받은 모든 클래스의 공통된 특징**

- 모든 nn.Module은 chile modules를 가질 수 있다. 
- 내 모델을 정의하는 순간, 그 모델에 연결된 모든 modules을 확인할 수 있다. 
	- 각 modules는 weights과 bias들을 가지고 있을 것이다. 
	- 학습을 진행하게 되면, predict값과 ground truth의 비교를 통해 loss를 발생시킨다. 
	- 이를 역전파로 전달하게되고, 이 loss로부터 다시 각 modules의 weights과 bias를 수정한다. 
	- 이때, 이러한 수정은 `.modules`를 통해 이루어질 것이다. 
- 이러한 과정을 pytorch는 용이하게 실행할 수 있다. 



**모든 nn.Module은 forward() 함수를 가진다.**

- 내가 정의한 모델의 forward()를 한번만 실행한 것으로 그 모델의 forward에 정의된 모듈 각각의 forward()가 실행된다. 
- 즉 쉽게얘기하면, nn.Module 상속을 통해 만들어진 모델 클래스는 자잘자잘한 모듈들의 forward의 실행을 통해 합쳐진다고 이해하면 쉽다. 



### Parameters

**모델에 정의되어 있는 modules가 가지고 있는 계산에 쓰일 Parameter**

아래의 방법들로 각각의 모듈들이 가지는 파라미터들(weights와 bias)을 Tensor로 출력해볼 수 있다. 

`sd = a.state_dict()` 

- 파라미터들의 키값과 tensor를 같이 출력해준다. 

`params = list(a.parameters())`

- 단순히 파라미터들의 tensor 값만을 출력해준다. 



**각 모델 파라미터들은 data, grad, requires_grad 변수 등을 가지고 있다.**

![image](https://user-images.githubusercontent.com/38639633/113479107-ef5a3000-94c7-11eb-8435-baf371f6425e.png)

- 이 각각의 파라미터는 텐서를 베이스로 가지는  하나의 클래스로 볼 수 있다. 
- 그렇기 때문에 여러가지 사용성 높은 변수를 가지고 있는데, `data`, `grad`, `requires_grad`가 그것이다. 
- `requires_grad` : boolean이기에 true vs false로 나타낸다. 
- `grad`는 각 파라미터가 어떠한 그래디언트를 가지고 있어서, 어떠한 미분값을 나타내게 되는지를 저장한 파라미터를 출력해준다. 
- requires_grad가 False로 되어있으면 해당 파라미터는 gradient를 업데이트하지 않는다. 
	- 이를 자주 활용하는 부분은 validation 및 test dataset에 대한 predict를 수행할 때 자주 사용한다. 

















