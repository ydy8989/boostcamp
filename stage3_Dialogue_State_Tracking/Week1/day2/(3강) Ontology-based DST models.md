# (3강) Ontology-based DST models

**강의 소개**

이번 강의에서는 지난 강의에서 DST 문제 풀이의 두 축 중 하나였던 Ontology-based method에 대해 알아보고자 합니다.

DST가 어떤 종류의 문제였는지를 다시 돌아보고, TOD에서 Ontology가 어떤 의미인지 자세히 알아봅니다.

또한, NBT, GLADE, GCE, SUMBT 등 Ontology를 이용해서 문제를 해결해 온 과정을 살펴보고, 

이러한 ontology-based approach가 어떤 장점과 한계점을 지니는지 배워봅니다.

이번 강의를 통해 DST를 푸는 두 축 중 하나를 자세히 이해해보고, 어떻게 구현할지 생각해보는 계기가 되길 바랍니다.



## Ontology

TOD/DST에서 Ontology란 무엇을 뜻하는지 알아봅니다.

### Classification based DST Recap

- Dialogue context와 slot을 각각 인코딩하는 모듈이 있고, 이 모듈들로부터 $V_t^j$의 확률값 $p(V_t^j\vert D_t, S^j)$를 구하는 것이 목표이다. 
- 이 때, $C_j= SET(S^j)$를 정의하고 j번째 slot의 확률을 구함으로써 원하는 $V_t^j$을 추론하는 과정이 classification 기반의 DST task라고 지난시간에 배웠다.

![image](https://user-images.githubusercontent.com/38639633/116190985-82ecec80-a766-11eb-90cb-54de97d9e3a3.png)

- 여기서 `빨간` 사각형 속 set $V^j$를 `Ontology`라고 한다



### Ontology

`Ontology`: 각 Slot $j$가 가질 수 있는 Value의 후보군을 정의해둔 정보

- Ontology-based model들은 이 ontology가 미리 정의되어 있고, 모든 value는 이 안에서만 등장한다는 가정에서 시작한다. 

![image](https://user-images.githubusercontent.com/38639633/116191342-19211280-a767-11eb-9941-5b9309f7f763.png)



### Ontology-based DST

- 사용된 Ontology가 실제 KB-Database에 존재함을 가정하고, 일종의 `Prior knowledge`로 활용할 수 있다!

![image](https://user-images.githubusercontent.com/38639633/116191521-6d2bf700-a767-11eb-86ca-d0a283d8ea7e.png)

> Recent Advances and Challenges in Task-oriented Dialog Systems (Zhang,et al 2020)



- Value에 대한 확률적 표현이 가능하다(단, 미리 정의된 ontology set 내에서)

	![image](https://user-images.githubusercontent.com/38639633/116191594-8cc31f80-a767-11eb-9d37-7153b0298a78.png)

- 하지만 다루는 Ontolgy의 volume이 커질 수록 computation cost가 증가한다. 또한, Unseen value가 등장했을 때 tracking하기가 어렵다. 



## Ontology-based DST Models

Ontology-based DST의 대표적인 모델들을 알아봅니다.

### Ontology-based DST Overview

- 각 slot마다 N개의 value라고 가정할때, 크게 두 가지 모듈이 존재한다. 
	- Encoder : Dialogue Context 등 인풋 인코딩
		- 다이얼로그와 slot j를 encoding하는 모듈
	- Scoring module : `P(Slot j=value n)` 계산
		- 자신이 속할 수 있는 value의 후보군들과 비교해서 얼마나 확률적으로 가능성이 있는지를 계산하는 모듈이다. 

![image](https://user-images.githubusercontent.com/38639633/116193805-e24cfb80-a76a-11eb-9e6e-d1de4811fc54.png)

### Neural Belief Tracker (NBT)

- 과거에는 DST를 heuristic하고 rule based하게 풀었다면, 이 모델은 data-driven으로 해결한 모델이다. 
- 아래 그림에서 `Candidate pair`라는 slot value들의 후보군을 볼 수 있다. 이 때 이 pair들과 context들을 인코딩, 게이팅 등을 하여 각 후보마다 binary decision을 진행하여 분류하게 된다. 

![image](https://user-images.githubusercontent.com/38639633/116194117-5ab3bc80-a76b-11eb-83af-d5efe41b7eee.png)



### GLAD

- 위와 비슷하게 encoder 모듈과 scoring 모듈을 지니고 있다. 
- 유저의 발화와 system action을 input으로 갖는다. 
	- DST가 아닌 Decision Making의 또다른 task인  Dialogue Policy에서 시스템이 할 수 있는 action이라는 것을 한정해놓고 action에 대한 것을 classification하는 것인데, 여기서는 input으로 사용하였다. 
	- 예) 가격에 대한 요청, 음식에 대한 요청
- **Slot value under consideration**
	- 우리가 고려하게 될 (score를 내고 싶은) ontology의 value 중 하나이다. 

![image](https://user-images.githubusercontent.com/38639633/116194147-64d5bb00-a76b-11eb-86ee-0cb298c46b61.png)

> Global-Locally Self-Attentive Dialogue State Tracker (2018, Zhong et al.)



모델을 조금 더 자세히 살펴보면

#### Encoding Module

일단 모델의 이름처럼 Global하고 locally하게 self attentive하게 들여다본다;

![image](https://user-images.githubusercontent.com/38639633/116194904-6f448480-a76c-11eb-9967-43054fa8b24d.png)

- 앞서 살펴본 전체 구조에서 encoder module을 살펴보면 위 그림과 같은 구조로 되어있다. 
- input이 들어오면 local한 BiLSTM와 global BiLSTM으로 이루어져 있다. 
- 여기서 $H^S$, $C^S$에서 $S$의 의미는 각 슬롯을 의미하며 위와 같이 각 슬롯별로 따로 BiLSTM과 self-attn layer를 갖고 있고, 이와 별개로 전체 Global BiLSTM과 self-attn을 mixture하며 진행하게 된다. 
- J(Slot-Specific)+1(Global) 개의 BiLSTM과 Self-attention weight 사용
- Slot과 고려 대상인 Value에 의존적인 인코딩
- 모든 Value candidates에 대한 binary classification
- **JxN encoding & Scoring** 을 사용한다. 
	- WOS 기준 46개의 LSTM사용;;



#### Scoring Module

- 위와 같이 각각 `system action`, `user utterance`, `slot value`를 encoding하게 되면 BiLSTM의 output인 Hidden state $H$, 그리고 Self-Attn의 ouput인 하나의 vector로 aggregation된 $C$가 나오게 된다. 

- `system action`, `user utterance`, `slot value`에 대해 각각을 encoding한 $H,C$는 아래와 같이 
	$$
	\begin{align}
	H^{utt}, c^{utt} &= encode(U)\\
	H_j^{act}, C_j^{act} &= encode(A_j)\\
	H^{val}, c^{val} &= encode(V)\\
	\end{align}
	$$
	의 형태로 나오게 된다. 

- 이후 아래와 같은 형태로 Attention module을 통과하여 계산하게 된다. 

	![image](https://user-images.githubusercontent.com/38639633/116196782-e0853700-a76e-11eb-98a7-51e13efac58d.png)

	



### GCE

- GLAD는 상당히 많은 수의 모듈들을 가지고 있었기에 scale up하기가 쉽지 않았다.
- Slot-Specific 했던 local encoder들을 하나로 통합sharing하였다. 
- 아래와 같이 관심 있는 하나의 slot type에 대해서는 토큰 형태로 추가하여 인코딩하기도 한다. 
- Slot dependent encoding!!

![image](https://user-images.githubusercontent.com/38639633/116197413-b2ecbd80-a76f-11eb-9ee3-caefff11d129.png)

> Toward Scalable Neural Dialogue State Tracking Model (2018, Nouri et al.)





## SUMBT

paper : [SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking.](https://arxiv.org/abs/1907.07421)

git : [https://github.com/SKTBrain/SUMBT](https://github.com/SKTBrain/SUMBT)

### Overview

- Slot, Value와 독립적인 encodings
- pre-trained LM 인코더 사용
- 두 종류 BERT 인코더
	- Dialogue Context Encoder
		- Trun-level hierarchical encoding
		- Trainable
	- Slot-value Encoder
		- Freeze(as Feature extractor)
- Slot-Utterance Matching Module
	- Turn utterances와 slot의 Fusion
- GRU based state Tracker 
- Non-parametric Scoring(Euclidean)

![image](https://user-images.githubusercontent.com/38639633/116199957-ae75d400-a772-11eb-83db-778105657e2d.png)

### Encoding module 1 : Dialogue context encoding

- Bert를 이용한 current turn[$r_t$, $u_t$] 인코딩 

- 현재 $t$ turn에서 시스템 발화와 유저 발화의 pair를 인코딩한다. 

	![image](https://user-images.githubusercontent.com/38639633/116200611-8044c400-a773-11eb-9161-a3909d3d1e91.png)

- output $U_t=BERT([X_t^{sys}, X_t^{usr}])$ 의 형태로 출력된다. 

- 총 n+m+3개의 사이즈로 이루어진 vector를 출력한다. 

	- n : 시스템발화 토큰 수
	- m : 유저 발화 토큰 수 
	- +3 : CLS 토큰 1개와 SEP 토큰 2개(중간, 끝)



### Encoding module 2 : Slot, Value Encoding

- [CLS] 토큰을 이용한 Pooling
	- bert는 cls 토큰을 통해 input 전체에 대한 vector representation을 진행함!
- 실제로는 Freezed encoder를 통해 미리 Ontology안의 모든 slot, value들을 pre-encoding 후 lookup한다.

![image](https://user-images.githubusercontent.com/38639633/116201300-50e28700-a774-11eb-86e9-af25cf463e17.png)



### Slot-Utterance Matching module

- Multi-head Attention을 통한 현재 턴 발화들에 대해 Slot s를 Query로 한 Aggregation한다. 

	- `Query`: Slot s vector $q^s$
	- `Key`: Current Turn t encoding $U^t$
	- `Value`: Current Turn t encoding $U^t$

	![image](https://user-images.githubusercontent.com/38639633/116201624-a61e9880-a774-11eb-800c-18333a3dde3b.png)

- multi-head attn layer를 통과하여 나온 hidden state vector $h_t^s$는 현재 turn $t$에 대해서만 attention을 가한 결과이다. 

- 그런데, 우리는 **모든** context $t$를 보고 decision을 해야한다. 



### GRU-based state Contextualizing

- GRU를 이용한 현재 Turn에서 Slot s에 대한 State를 Context turns State에 대하여 Contextualize한다.

	- `Input`: $h^s_t$ == Slot $s$를 Query로 하여 Turn $t$를 aggregation한 vector
	- `Output`: $d^s_t$ == 이전 Turn ($t-1$)에서 Slot $s$에 대한 hidden state와 gating을 통해 update된 현재 Turn hidden state
	- Layer Normalization을 통해 최종 현재 Turn $t$에서 Slot $s$에 대한 representation

	![image](https://user-images.githubusercontent.com/38639633/116202179-44aaf980-a775-11eb-836e-ae161d5470a1.png)

- 이 과정을 거치면 slot type $s$에 대하여 현재 turn $t$까지의 context 정보가 집약된 $\hat{y}_t^s$가 나오게 된다. 



### Non-Parametric Scoring

![image](https://user-images.githubusercontent.com/38639633/116202533-a53a3680-a775-11eb-8b59-620327255ca5.png)

- 이렇게 나온 $\hat{y}_t^s$은 미리 encoding 해둔 value에 대한 representation vector $y_t^s$와의 비교를 통해 score를 계산한다. 
- Distance function (Cosine Similarity, Euclidean Distance)을 이용한 **Dialogue Context을 고려한 Slot $s$ vector** $\hat{y}_t^s$ 와 **Value under consideration vector** $y_t^s$ 간의 Scoring을 진행



### Training Critera

$t$-turn 전까지의 모든 **시스템 발화**와 **유저 발화**가 주어졌을 때, t-turn의 value가 나올 확률은 다음과 같다.
$$
p(v_t\vert \mathbf{x}_{\leq t}^{sys}, \mathbf{x}_{\leq t}^{usr}, s) = \frac{exp(-d(\hat{y}_t^s, y_t^v))}{\sum_{v'\in \mathcal{C}_s}exp(-d(\hat{y}_t^s, y_t^{v'}))},
$$

- Slot $s$에 속하는 모든 $v$에 대한 scoreing을 통한 Normalization을 진행한다. 
- 여기서 $\mathcal{C_s}$는 `Ontology`가 된다.



optimization은 다음과 같이 진행한다.
$$
\mathcal{L}(\theta)=\sum_{s\in \mathcal{D}}\sum_{t=1}^{T}\text{log}p(v_t\vert \mathbf{x}_{\leq t}^{sys}, \mathbf{x}_{\leq t}^{usr}, s)
$$
최종적인 모든 object는 $t=1$부터 마지막 턴 $T$까지 구하게 되고 이때, slot type $s$는 $\mathcal{D}$에 속한다. 여기서 말하는 $\mathcal{D}$는 미리 define 되어있는 `Slot meta`이다. 



### Evaluation (Single-Domain)

**BERT-RNN:**

- Dialogue Encoder와 GRU State Tracker
- Slot-Value (Ontology) 갯수만큼의 dimension으로 Projection 후 Classification

**BERT-RNN-Ontology:**

- Dialogue Encoder와 Ontology Encoder
- Slot-Value level의 Matching
- GRU State Tracker
- Classification

**Slot-dependent SUMBT:**

- SUMBT와 동일하지만 Slot마다 따로 학습

*$\Rightarrow$  BERT-RNN과 BERT-RNN-Ontology는 Ontology가 Update되면 재학습이 필요 (Not Scalable)*

![image](https://user-images.githubusercontent.com/38639633/116219719-dff89a80-a786-11eb-911d-2e5fc11f6e23.png){: width = "70%"}



### Evaluation (Multi-Domain)

 ![image](https://user-images.githubusercontent.com/38639633/116224323-7f1f9100-a78b-11eb-9c6b-4e8620a3f5b9.png)



**Attention weight analysis**



![image-20210427190541110](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210427190541110.png)

**Further Reading**

- [SUMBT paper](https://www.aclweb.org/anthology/P19-1546/)