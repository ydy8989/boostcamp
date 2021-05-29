# (1강) DKT 이해 및 DKT Trend 소개

**강의 소개**

Deep Knowledge Tracing (DKT) 첫 강의에 오신 것을 환영합니다! 이 강의는 DKT에 관해 전반적인 overview를 제공함으로서 여러분들이 DKT라는 생소한 세계에 친숙해질 수 있도록 도와줄 것입니다! 각 내용은 무겁지 않고 필요한 요소들을 전달하는 것에 초점을 두고 있으니 가벼운 마음으로 따라가주시면 됩니다! 

우리는 DKT가 무엇인지 그리고 왜 필요한지 먼저 이해를 할 것입니다. 이후 DKT 대회가 어떻게 진행되는지, 중요하게 보는 Metric이 무엇인지 이해하고, DKT의 역사와 트렌드를 살펴보게 될 것입니다. 최종으로 DKT가 sequence 데이터를 사용하는 만큼 DKT에서 사용되는 sequence 모델의 발전 과정을 리뷰할 것입니다.

DKT에 관해 소개하는 첫 강의이자 마지막 강의로 이후 강의부터는 대회와 관련된 유익한 정보들과 꿀팁에 초점을 두어 진행되니, DKT 그 자체를 더욱 깊게 파보고 싶으신 분들은 아래 Further Reading에 적혀진 논문에서 출발을 추천드립니다. 만약 더 멀리 DKT 내부로 탐험을 하고 싶으신 분이 계시다면, 1강 참고자료로 제공되는 history of deep knowledge tracing 자료의 DKT의 발전과정을 참조하여 공부하면 좋을 것 같습니다.

 

**Further Reading**

[2015 DKT 논문](https://arxiv.org/pdf/1506.05908.pdf) - DKT에 딥러닝이 적용된 최초의 논문입니다! LSTM을 이용해 모델링했으며, 지식 상태를 어떻게 표현하고 실생활에 활용하는지 가볍게 훑어보세요!



## DKT?

"Deep Knowledge Tracing : 딥러닝을 이용한 지식 상태 추적"을 의미한다. 



**그렇다면 Knowledge tracing은 뭘까?**

![image](https://user-images.githubusercontent.com/38639633/119290645-ea278f00-bc87-11eb-9a21-4dc0d4fc33d6.png)

- 초등학생의 사칙연산 이해도를 알아보기 위해 시험을 보았다고 가정하자

- 이때, 더하기, 빼기, 곱하기, 나누기와 같이 우리가 학생의 시험을 통해 알고싶은 지식을 `지식 구성요소`라고 한다.

- 시험문제를 통해 파악한 각 지식에 대한 학생의 이해도를 `지식 상태`라고 한다.

	![image-20210524120445971](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210524120445971.png)

- 중요한 점은 **지식의 상태는 계속 변화한다는 점이다**

	- 그러므로 변화하는 지식상태를 계속 추적해야만 한다.



**DKT 원리**

- 시험지를 통해 예측한 지식의 상태로부터 각 지식 구성요소의 분포를 파악하고, 지식의 상태를 예측할 수 있게된다.

- 이때, 문제 풀이 정보가 추가될수록 학생의 지식 상태를 더 정확히 예측할 수 있다.

	![image](https://user-images.githubusercontent.com/38639633/119291002-aaad7280-bc88-11eb-9a7d-f11560948a52.png)

- 이렇게 수렴된 분포로부터 다음 문제를 예측하게 된다.

	![image](https://user-images.githubusercontent.com/38639633/119291050-c31d8d00-bc88-11eb-9d27-1f07eb224a31.png)

- 당연하게도 데이터가 많을수록 지식 상태 예측은 더욱 정밀해진다. 



**DKT 대회**

- 대회에서는 지식상태보다는 주어진 문제를 맞췄는지 여부에 집중한다. (Binary classification)

	![image](https://user-images.githubusercontent.com/38639633/119304302-06382a00-bca2-11eb-8fef-9d8af7c8a801.png)

- 당연하게도 TEST 셋에서의 맨 마지막 정답 여부는 주어지지 않고, 이를 예측하는 것이 대회의 목적이다. 

	![image](https://user-images.githubusercontent.com/38639633/119305755-223ccb00-bca4-11eb-98d6-5fcf35bac464.png)



## Metric의 이해

### AUC / ACC

- 강의자료로 대체



## DKT History

![image](https://user-images.githubusercontent.com/38639633/119307910-549bf780-bca7-11eb-8a20-8d2feeb4e4d9.png)

- Graph neural network를 이용한 dkt가 가장 최근의 모델이다
- 하지만 대회에서 좋은 성능을 보이는 것들은 trasnformer를 이용한 방법이다.



## DKT Trend

- DKT는 Sequence Data를 다루는 만큼 자연어 처리 분야의 발전에 많은 영향을 받아왔다. 
- 자연어 처리 분야에서 다양한 아이디어들이 쏟아져나오고 있지만, DKT 분야로의 접목은 그 속도가 더디다
- 다른 방향에서 생각하면 아직 시도할 것들이 많은 만큼 무궁무진한 발전 가능성이 있다는 의미이다. 
- 최근에는 자연어 처리 말고도 이미지, 그래프 모델 등의 다양한 접목이 이루어지고 있다. 

**MODEL**

- Transformer 위주의 모델링이 대세를 이루고 있으면, 최근 GNN과 CNN을 활용하고자 하는 시도가 있었다. 
- `LSTM` `MANN` `GCN` `Trnasformer` `Bi-LSTM` `GNN` `Attention` `CNN`

**DATA**

- 과거에 모델링에 활용하던 데이터 feature의 수가 많지 않았었다. 최근들어 Elasped time, Lag time 등과 같은 다양한 feature들을 활용하기 시작했다
- `Question ID` `Question Category` `Text Description` `Response` `Timestamp` `Elasped time` `Lag time`

**REGULARIZATION TERM**

- 기존 모델의 성능 개선을 위해 구조를 변경하지 않고 정규화 항을 추가하여 개선시키려는 시도가 종종 눈에 띈다. 
- `Reconstruction` `Waviness` `Skill`

**EMBEDDING**

- 별다른 탐구가 이루어지지 않는 영역이며 과거 방식이 답습된다. 
- qDKT의 fasText 활용은 아직 이곳에 발전의 여지가 있음을 보여준다.
- `Category` `Position` `fastText`



