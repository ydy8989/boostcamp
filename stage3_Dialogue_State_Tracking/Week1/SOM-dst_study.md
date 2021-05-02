# Efficient Dialogue State Tracking by Selectively Overwriting Memory(ACL 2020)



### SOM-DST Model contribution

- DST task를 2개의 task로 나누어 효과적으로 slot에 해당하는 value 생성
	- Task1 : State operation prediction(Encoder)
		- slot에 해당하는 value를 generation해야하는데, 이 각각의 slot에 값이 할당될것인가 vs 그렇지 않은가를 먼저 찾아내기 위해서 task1 정의
		- slot을 찾기 위해서 state operation prediction을 인코더 단에서 진행 
	- Task2 : Slot value generation(Decoder)
		- 찾아낸 slot에서 value를 generation함
- Selectively Overwriting Memory
	- 기존 모델 대비 효율적인 DST Model 구축
	- Slot의 minimal subset(selective slot)만 가지고 value 생성 가능함 
- 논문 제출 시기 기준 SOTA 달성



### SOM-DST

- State Operation Predictor와 Slot Value Generator로 구성되어있음

- Definition

	- Dialogue State

		- $t, \mathcal{B}_t=\left\{(S^j, V_t^j)\vert a\leq j\leq J\right\}$
		- $t$ : Dialogue turn
		- $\mathcal{B}_t$ : Key가 slot인 fixed-sized memory
		- $S^j$ : Slot
		- $V_t^j$ : Slot value
		- $J$ : Total number of such slot (각 도메인의 slot은 미리 정해져 있음(혹은 있다고 가정))

	- Special Value

		- 각 slot에 할당된 Value들 중 특별한 의미를 가지는 Value.
		- `NULL` : Slot에 아무런 정보가 없어서 Value도 없는 상태(초기값으로 주어짐)
		- `DONTCARE`: 해당 대화에서 신경 쓰지 않아도 되는 상태 (Tracking 대상에서 제외)

	- Operation

		- $r_t^i\in \mathcal{O}=\left\{CARRYOVER, DELETE, DONTCARE, UPDATE\right\}$

		- 하나의 Dialogue turn마다 state operation predictor를 통해 선택되는 값

		- 한 개의 Slot마다 하나의 Operation이 설정된다. 

			- `Slot` ---`Operation`-->`Value`

		- Operation 에 따라 Value가 설정된다.
			$$
			V_t^j=\begin{cases}
			V_{t-1}^j &\text{if } r_t^j=CARRYOVER\\
			NULL &\text{if } r_t^j=DELETE\\
			DONTCARE&\text{if } r_t^j=DONTCARE\\
			v&\text{if } r_t^j=UPDATE
			\end{cases}
			$$

- State Operation Predictor : Classification task를 수행하는 Encoder(bert)

	- 인코더 부분 : BERT 모델로 구성
	- Input : Dialogues and Previous dialogue states
	- Output : 각각의 slot마다 output이 있다고 생각하면 됨 
		- 각각의 slot은 위의 4가지로 이루어짐
	- Previous dialogue states의 slot 하나하나마다 이 값들이 부여된다.

	

