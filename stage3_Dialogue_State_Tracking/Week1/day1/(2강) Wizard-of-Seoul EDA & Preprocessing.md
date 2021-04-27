# (2강) Wizard-of-Seoul EDA & Preprocessing

**강의 소개**

이번 강의에서는 강의를 통틀어 사용할 한국어 Dialogue State Tracking 데이터인 **Wizard-of-Seoul**에 대해 소개합니다.

WoS 데이터셋의 형태와 을 다루기 위한 방법을 소개하고, DST의 평가방법과 구현 방법을 소개합니다.

오늘 소개하는 WoS 데이터셋을 살펴보며 대화 데이터가 어떻게 구성되었고, 어떻게 다루는지 익혀보도록 해요!

 

## introduction to Wiard of Seoul

### Overview

**multi-domain dialogue state Tracking**

- `MultiWOZ`는 TOD의 가장 대중적인 벤치마크 데이터셋
- (영국)Cambridge 여행 Assistant 시나리오(총 7가지 멀티 도메인)
- 멀티도메인 DST는 도메인 간의 전이 상에서 나타나는 Lexical Entailment, Co-referencing 등을 해결하는 것이 주요 Challenge이다. 

- Wizard of Seoul(WOS) 는 KLUE Project의 일환으로 새롭게 구축된 한국어 DST 벤치마크
- 유사하게 서울 여행 Assistant 시나리오(총 5가지 멀티 도메인)로 구성되어 있다. 
	- Annotation interface를 개선하여 Typos 등 Error types 해소
	- 많은 수의 Boolean type slot을 포함하고 있어 abstractiveness 극대화
	- Counterfactual Goal을 사용하여 내재한 Bias를 최소화하였다. 



###  Multi-Domain Dialogue State Tracking

- Domain Transition 에 의해 abstractive하게 발생하는 state변화를 감지해야 하며, 이에 따라 context에서 적절한 value를 co-referencing해서 Tracking해야 한다. 

![image](https://user-images.githubusercontent.com/38639633/116174515-8b82fa00-a749-11eb-9927-9f1f05679cb1.png)

![image](https://user-images.githubusercontent.com/38639633/116174530-90e04480-a749-11eb-8bba-94686cbbe59d.png)

대화를 진행하면서 정해진 value로부터 referencing을 tracking한다. 



### Data Construction

Wizard of Oz 라는 framework로부터 데이터가 구성되었다. 

![image](https://user-images.githubusercontent.com/38639633/116174839-2bd91e80-a74a-11eb-85b9-4df7c287c2cc.png)

- 유저는 좌측과 같은 instruction을 받게된다. 
- informable slot과 requestable slot으로 이루어진 context를 합쳐서 자연어로 풀어서 쓴 것을 `goal instruction`이라 부른다. 



User Role을 맡은 worker는 주어진 goal instruction에 따라 대화를 이어나가도록 요청 받는다. Goal Instruction은 유저가 달성해야 할 목적 (User Goal)이 유저의 Context, Informable Slot과 Requestable Slot을 포함하여 자연어로 기술 되어 있다

![image](https://user-images.githubusercontent.com/38639633/116175145-bd489080-a74a-11eb-870f-084fcd07e338.png)



### Data Format

- Json Format - Each examples consists of information below:
	- dialogue_idx - str : unique id for a dialogue
	- domains - List[str] : single or multi domains for the dialogue
	- dialogue - List[Dict] : turn-level information that contains
	- attributes below:
		- role - str : a speaker of the turn, user or sys
		- text - str : an utterance (transcription)
		- state - List[str] : dialogue-state formed in DOMAIN-SLOTVALUE

![image](https://user-images.githubusercontent.com/38639633/116175462-4eb80280-a74b-11eb-977b-3e6efb203d6c.png)



## Taxonomy of DST models

DST model을 구현하는 두 가지 방법에 대해 소개합니다. 



### Recap

objective는 다음과 같다. 
$$
\prod_{t=1}^{T}\prod_{i=1}^{J}p(V_t^j\vert D_t, S^j)
$$

### Model Concept

여기서 $t$ 번째 턴으로 고정한 채 살펴보면
$$
\prod_{i=1}^{J}p(V_t^j\vert D_t, S^j)
$$


으로 고정되고, 이 때의 구조를 도식화해보면 다음과 같다.

![image](https://user-images.githubusercontent.com/38639633/116181714-110ca700-a756-11eb-94c3-62a2eb9f5ff4.png)

- Dialogue context를 인코딩하고, Slot meta를 인코딩하여 recover state라고하는 $B_t$를 추론하여 이를 예측에 활용할 수 있을 것이다. 
- 여기서 더 나아가 J term도 고정시키면 다음과 같다. 

![image](https://user-images.githubusercontent.com/38639633/116182240-dd7e4c80-a756-11eb-8093-267372591b1e.png)

- 이처럼 우리는 input으로 받은 $D_t$와 '관광-종류'라는 Slot일때 '박물관'이라는 정답이 나올 확률을 모델링해야한다. 여기서 두 가지 방식의 모델링이 존재하는데 하나는 `classification`이고 또 다른 하나는 `Generation`이다.



### How to represent p(V)?

**Classification**

- $j$번째 Slot $j$에 속할 수 있는 모든 Values의 집합을 정의(가정)한다.
	$$
	V^j\in Set(S^j) = \left\{V^{ji}\vert 1\leq i\leq \vert Set(S^j)\vert\right\}
	$$

- 이후 $p(V_t^j\vert D_t, S^j)$를 아래와 같이 Classification할 수 있을 것이다.

	![image](https://user-images.githubusercontent.com/38639633/116183110-49ad8000-a758-11eb-9b0b-de100969f8dd.png)

	![image](https://user-images.githubusercontent.com/38639633/116183134-53cf7e80-a758-11eb-8bc2-438eca63c05b.png)



**Generation**

- Conditional Lanuage modeling으로 formulation, vocab space상에 표현한다.
- ![image](https://user-images.githubusercontent.com/38639633/116183269-8d07ee80-a758-11eb-9dae-a9fd993778ae.png)



### Taxonomy of DST models

여기서 Classification based 모델을 `Ontology-based DST`라고 하고, Generation / Extraction 방식의 모델링 방식을 `Open-vocab based DST`라고 한다. 

 





**Further Reading**

- MultiWOZ (영어 DST 데이터)
	- [MultiWOZ 2.0 Github](https://github.com/budzianowski/multiwoz)
	- [MultiWOZ 2.4 paper](https://arxiv.org/abs/2104.00773)

