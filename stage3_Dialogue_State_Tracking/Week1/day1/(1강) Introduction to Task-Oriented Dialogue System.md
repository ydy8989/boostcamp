# (1강) Introduction to Task-Oriented Dialogue System

**강의 소개**

DST 첫 강의에 오신 모든분들 환영합니다 😎!!

드디어 본격적인 Dialogue State Tracking 강의를 시작합니다 🔥🔥

사람과 시스템이 서로 소통하기 위한 수단으로써 대화를 이해하는 것은 매우 중요하며, NLP의 주요한 한 분야로 자리잡고 있습니다.

이번 시간에는 대화 태스크의 한 부분인 Task-Oriented Dialogue를 소개하고, 이 태스크에서 해결할 문제가 무엇인지를 알아보고자 합니다.

특히, TOD를 이해하는 방법인 Dialogue State Tracking이 무엇인지, 그리고 이 문제를 해결하기 위해 가정하고 있는 점과 제약 사항 등에 대해 알아봅시다. 이 강의를 통해 딥러닝이 대화를 이해하는 방법을 공부해보도록 해요!

 

## (Task-Oriented)Dialogue System(TOD)

### Taxonomy of dialogue system

크게 두가지로 분류된다. 

**Task-Oriented dialogue & Open-domain dialogue(Chit-Chat)**

![image](https://user-images.githubusercontent.com/38639633/116138577-f4e61700-a70f-11eb-9de9-5e686fdea242.png)

- Taks-Oriented dialogue
	- 특정 태스크의 수행을 성공시키는 것이 목적인 다이얼로그이다. 
- Open-domain dialogue
	- 흔히 말하는 잡담이나, 대화 주제가 특정되어있지 않은 형태의 다이얼로그를 의미한다.



여기서 우리가 관심있는 부분은 task oriented dialogue이다. 

![image](https://user-images.githubusercontent.com/38639633/116139647-34f9c980-a711-11eb-800e-6e7c27885932.png)

미리 정의된 시나리오에 대한 task이며, 다음으로 predefined scenario에 대해 알아보자



### Task-oriented dialogue : Problem definition

task oriented dialogue as information exchange game : 일종의 게임 같은 것이다. 

유저는 자신의 목적을 말하고, 시스템은 knowledge base로부터 정보를 추출하여 유저가 원하는 목적을 달성할 수 있게하는 것이 일련의 과정이라고 말할 수 있다. 

![image](https://user-images.githubusercontent.com/38639633/116139323-dcc2c780-a710-11eb-8a33-226bf866ed77.png)



여기서 **Predefined Scenario** 는 어떻게 정의되고 구성되는 것인가?

![image](https://user-images.githubusercontent.com/38639633/116139742-4f33a780-a711-11eb-8488-a8a6f1a00676.png)

- User Goal
	- 시스템에 접근해서 원하는 정보를 찾거나, 자신이 가진 추가적 정보를 제공해서 새로운 정보를 예약 혹은 trasnsaction을 일으키는 것을 가정으로한다. 
- Task Schema
	- 유저가 원하는 정보를 찾거나 시나리에 맞는 정보를 줄 수 있도록 정의 된 메타 정보를 의미한다. 
- Task related
	- 시나리오에서 제공하고자하는 정보 DB



### TOD : Task Schema

![image](https://user-images.githubusercontent.com/38639633/116140328-24961e80-a712-11eb-94fc-e72d272fa39a.png)

이렇기에 Task schema는 우리의 task에서 굉장히 중요한 역할을 갖는다. 

- TOD는 user goal의 파악 및 연계된 task의 성공이 목적
- user goal은 크게 2가지 종류의 정보로 구성 된다고 가정한다.
	- `informable slot` : 특정 kb instance를 찾거나, 새로운 instance를 write하기 위해 user가 system에게 주거나 맥락에 의해 user가 의도할 수 있는 타입의 정보(대화에 대한 제약 사항 및 DST의 target)
	- `Requestable slot` : 특정 KB instance가 선택된 이후, 추가로 정보를 요청할 수 있는 타입의 정보(system이 user에게 제공)
- 이러한 정보의 '정의'가 바로 Task Schema



#### Task Schema 예시

![image](https://user-images.githubusercontent.com/38639633/116140623-8eaec380-a712-11eb-873f-6b0d74964610.png)

- 결국 유저가 알고있는 약간의 정보를 제공하면서 물어보는 informable과 그렇지 않은 requestable로 이루어져있다고 할 수 있다. 



최근에는 Schema-Guided Dialogue Dataset이라는 더 풍부한 메타정보를 담고 있기도 하다. 

![image](https://user-images.githubusercontent.com/38639633/116140958-067cee00-a713-11eb-8089-2de57399b69d.png)

> [https://github.com/google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)



### Knowledge Base

knowledge base는 이 시나리오를 성공시키기 위해서 시스템쪽에서 접근할 수 있는 정보를 의미한다. 
Knowledge base는 앞서 다룬 Task schema(시나리오)를 따른다. 

![image](https://user-images.githubusercontent.com/38639633/116141311-768b7400-a713-11eb-9d0b-0a043dcc0438.png)

- Task schema와 연계되는 system입장에서 접근가능한 structured KB
- system은 특정 instance를 찾거나 새로운 instance를 생성하기 위해 informable slot을 tracking해야 한다.
- 특정 instance가 주어진 이후에는 요청 받은 requestable slot의 value를 답변할 수 있다. 
	- 즉, structred KB는 앞선 TAsk schema를 따른다.



### TOD 예시

![image](https://user-images.githubusercontent.com/38639633/116141636-e7329080-a713-11eb-9156-bc3832247575.png)

- 파란색은 informable slot을 의미하고, 초록색은 requestable slot을 의미한다. 



### Components of modularized TOD system

![image](https://user-images.githubusercontent.com/38639633/116141934-485a6400-a714-11eb-9be8-1704d4c45e13.png)

- tod system은 위와 같은 방식으로 진행되며, 우리는 DM(Decision Making) 파트의 DST 모듈에 대한 내용을 공부할 것이다. 



## Dialogue State Tracking

### Dialogue State(Tracking)

**Dialogue state**: (Task schema에 의해) 미리 정의된 informable slot 타입 J개에 대해 user가 의도하고 있는 valeu의 상태 즉 j개의 slot, value pair의 set으로 표현될 수 있다. 
$$
B_t = \left\{(S^j, V_t^j), 1\leq j\leq J\right\}
$$
![image](https://user-images.githubusercontent.com/38639633/116142465-e6e6c500-a714-11eb-8d06-99b8434e612f.png)

- 이 정보를 유저가 의도하고 있는 정보의 상태를 매 턴마다 tracking하는 task라고 할 수 있다. 
- 여기서 $B_t$는 $t$번째 dialogue turn에서의 유저가 원하는 informable slot과 value의 pair를 의미한다. 
- 따라서 위 예시에서 $B_1$에서 $B_4$로 갈 수록 점점 **중첩**되는 것을 확인할 수 있다. 
- 하지만 사실은 J개의 slot을 모두 추적해야만 한다. 즉, 위의 예시는 생략되어있는 예시이고 사실은 아래와 같다.

![image](https://user-images.githubusercontent.com/38639633/116152273-a9883480-a720-11eb-82f3-c0f23ecff77c.png)

- 위와 같이 사실은 `none`으로 되어있고 이는 생략이 되어있다. 
- 주의할 점은 2가지 special case를 위한 values들이 있다.
	- `none` : 현재까지 대화에서 해당 정보가 아직 의도되지 않았거나 취소된 상태.즉, 대화 시작 시 모든 Slot은 Value로 none을 가지고 있는 상태, 예시에서는 편의 상생략하였다. 애초부터 없거나 대화 진행 중 value가 회수되는 상태도 여기에 해당한다.
	- `dontcare` : 해당 slot에 어떤 value가 와도 상관 엇어서 검색 조건에서 제외되는 상태



이제 시스템의 발화와 유저의 발화를 각각 $r_j$와 $u_j$로 나타내보자. 이때, Dialogue Context $D_t$는 다음과 같다.
$$
D_t=(r_1,u_1,r_2,u_2,\dots, r_t, u_t)
$$

- input: t-번째 턴까지의 Dialogue context
	- 한 턴은 각각 system response r 과 user utterance u로 이루어져있다. 

이 때의 output은 t-번째 턴까지 추정되는 Dialogue state이다.

이를 수식적으로 objective로 표현하면 
$$
\prod_{t=1}^{T}\prod_{i=1}^{J}p(V_t^j\vert D_t, S^j)
$$


와 같이 표현될 수 있다.



### Evaluation Metrics

2가지 정도의 metric이 있다. 

- `Joint Goal Accuracy(JGA)` : 예측한 $B_t$와 Ground Truth $B_t$ 사이의 Exact Matching
- `Slot Accuracy(SA)` : $B_t$간의 비교가 아닌 개별 Slot j의 pairs $(S^j, V_t^j)$ level의 Accuracy



**Example)**

J의 갯수가 4라고 가정할 때(Slot A, B, C, D)

![image](https://user-images.githubusercontent.com/38639633/116155596-43ea7700-a725-11eb-9abf-e0d4359c313d.png)

- 여기서 JGA의 경우 하나만 틀려도 0점인 매우 어려운 metric이다. 
- SA는 열마다 일치하는지의 여부를 count하여(=4) 4로 다시 평균을 낸 점수(=1)이다. 



![image](https://user-images.githubusercontent.com/38639633/116167189-ad28b500-a73a-11eb-86fb-06ce420a9db3.png)

이 경우 metric은 다음과 같다.

- JGA : 0(Slot A에서 하나 틀렸기 때문에 나머지 모두 맞더라도 0점)
- SA는 열마다 일치하는지의 여부를 count하여(=3) 4로 다시 평균을 낸 점수(=0.75)이다. 



이처럼 JGA는 매우 어려운 metric이고, 실제로는 SA를 더 많이 사용하는 편이다.



## Properties of DST

쉽게 떠올릴 수 있는 Slot Value Extraction approach와의 비교를 통해 DST라는 Task가 가지는 속성을 알아본다.



### Slot Extraction vs DST

![image](https://user-images.githubusercontent.com/38639633/116167412-1c060e00-a73b-11eb-954f-ac03d744ce5f.png)

- 가장 큰 차이는 slot에 대한 goal이 다르다. 
- NLU의 경우, Slot value를 추출하는 것이지만, DST의 경우, 미리 **predefined**된 slot value를 추측하는 데 그 목적이 있다. 
- abstractive slot value에 대한 추론이 가능한가? 
	- DST는 boolean의 형태로 이같은 추론이 가능하다. 
	- 예를들어 문장에서 "인터넷이 가능한가?"에 대한 내용을 추출한다고 가정할 때, 어느 토큰까지를 해당 내용의 추론 범주인지 정의하기가 힘들다
	- 반면에 DST는 예/아니오 등으로 명시적으로 나타내기 쉽다. 
- 이 같은 점이 NLU와의 차이점이라고 할 수 있다. 
	- 단순히 dialogue context에서 드러나는 정보들을 추출하고 파악하는데에 그친다면, DST는 대화 정책상의 정보들을 집계하고 보정하고 취합하는 큰 범위의 일을 수행한다고 할 수 있다. 



### DST as Entity Extraction

**Using BIO tags**

 ![image](https://user-images.githubusercontent.com/38639633/116168106-ac911e00-a73c-11eb-8b53-bdcd56902c14.png)

- 위와 같이 BIO 태그를 통해 예측하는 것은 매우 어렵다. 
- 애매모호한 내용의 slot의 BIO 범주를 태깅하는 것은 난해하기 때문이다. 
- 이는 Boolean type의 slot이나 dontcare나 none과 같은 special한 case를 처리하기 위한 추가적인 후처리가 필요하다.



### Entity typing & identification

![image](https://user-images.githubusercontent.com/38639633/116168399-3ccf6300-a73d-11eb-8c6c-21c1db292bea.png)

- Entity의 typing과 Identification을 동시에 해야하는 문제이기도 하다. 
	- typing : 여덟명 / 숙소 - 예약 명수
	- identification : 여덟명 -> 8

위의 인터넷과 같은 예시에서도, 대표값으로 표준화가 진행되어야만 한다. 인터넷이 가능하다는 표현은 여러 형태로 나타날 수 있기 있으며, 이를 모두 `yes`라는 표준값으로 통일해야하는 후처리가 진행되어야만 한다. 



### Decision making based on Context

Slot에 대한 판단이 여러 턴에 걸쳐 결정되기도 한다. 

![image](https://user-images.githubusercontent.com/38639633/116168609-c717c700-a73d-11eb-8fbe-5520128de493.png)

### State Update

매 턴마다 Slot의 value를 추론하기 때문에, value의 변화 즉, state의 update도 담당해야만 한다.

![image](https://user-images.githubusercontent.com/38639633/116168681-f9292900-a73d-11eb-8fad-d8aa97a29057.png)

'적당'이나 '비싼' 모두 가격대에 맞는 value로 뽑힐텐데, 조금 더 복잡 미묘하게 state update가 일어날 수 있기에 이러한 작업도 감당해야하는 task이다. 



**Further Reading** 

- [Recent Advances and Challenges in Task-oriented Dialog Systems (TOD 서베이 논문)](https://arxiv.org/pdf/2003.07490.pdf)
- [Tutorial : Deeper Conversational AI (Neurips)](https://neurips.cc/media/Slides/nips/2020/virtual(07-08-00)-07-08-00UTC-16657-track2_deeper.pdf)