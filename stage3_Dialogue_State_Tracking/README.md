🤩**Dialogue State Tracking** 과정에 입과하신 것을 환영합니다! 🤩

본 페이지는 Task-Oriented-Dialogue의 핵심적인 테스크 중 하나인 Dialogue State Tracking의 소개와 핵심적인 모델 아키텍처를 소개합니다. 또한 테스크 내에 내재한 여러가지 챌린지(Scalability, Robustness)들을 함께 고민해보고 이를 해결하기 위한 노력들을 배웁니다. 



### [WEEK 1](https://github.com/ydy8989/boostcamp/tree/main/stage3_Dialogue_State_Tracking/week1)

**[4/26] Introduction to DST & WoS Data EDA**

- 1강 : 

	 Introduction to Task-Oriented Dialogue System

	- TOD 소개
	- 실습 : Starter Code

- 2강 : Wizard-of-Seoul EDA and Data processing

	- 한국어 DST 데이터셋인 WOS 소개
	- 실습 : WoS 데이터 EDA , 전처리 code 

- Daily Mission

	- WoS data EDA 해보기 !
		- Domain 갯수 / Slot 갯수 / Slot당 Value 갯수 확인해보기
		- Domain별 자주 사용되는 Bigram, Trigram 확인해보기

 

**[4/27] Ontology-Based DST Model**

- 3강 : Ontology-based DST models
- 4강 : Ontology-based DST model (SUMBT) 실습
- Daily Mission
	- WoS를 로딩하여 SUMBT의 input feature 만들어보기
	- SUMBT 모델 학습에 필요한 Preprocessor 완성하기
		- convert_example_to_feature
		- recover_state

 

**[4/28~30] WEEK 1 Special Mission** 

 

**[4/28] Special Mission 1 : SUMBT-Notebook 공개**

- 코드 빈칸을 채워 SUMBT 완성하기
- Ontology Pre-encoding 진행하기 (BERT_SV)

 

**[4/29, 4/30] Special Mission 2 : SUMBT 학습**

- 주어진 코드를 기반으로 SUMBT 학습 하기
- 추론하여 리더보드에 Submission하기

 

### **[WEEK 2](https://github.com/ydy8989/boostcamp/tree/main/stage3_Dialogue_State_Tracking/week2)** 

 **[5/3] Open-Vocab Based DST Models**

- 5강 : Open-Vocab Based DST models (이론)
- 6강 : Open-Vocab Based DST models 실습
- Daily Mission
	- TRADE 모델에 필요한 전처리 모듈 만들어보기
	- TRADEPreprocessor
		- convert_example_to_feature
		- recover_state

**[5/4] Other DST Models**

- 7강 : Hybrid Approach
- 8강 : Prediction & Generation
- Daily Mission
	- WoS 데이터의 slot type을 구분해봅시다
	- 주어진 데이터의 Slot중에서 Categorical slot과 picklist slot type 구분하기
		- Boolean Type

 

**[5/4] 오피스아워 (이유경 멘토)**

- SUMBT 모델 학습코드 제공
- Huggingface transformers Recap
	- Trainer Customization
	- Requirements 고려 필요함

 

**[5/5] 어린이날 휴강** 

**[5/6~7] WEEK2 Special Mission**

 

**[5/6] Special Mission 1 : TRADE 모델의 Slot Gate 변경해보기**

- Gate 3개 ⇒ 5개로 바꾸면 성능이 올라감
	- none, ptr, dontcare ⇒ none, ptr, dontcare, **yes, no** (5개)
- word dropout 적용해보기

 

**[5/7] Special Mission 2 : TRADE 모델의 Encoder 변경해보기**

- Encoder를 GRU에서 Transformer encoder (PLM)으로 바꿔보기
- max_seq_length 핸들하기

 

**[5/7] 오피스아워 (박규민 멘토)**

- TRADE 베이스라인 설명
- 강의 및 리더보드 관련 질의응답

 

### [WEEK 3](https://github.com/ydy8989/boostcamp/tree/main/stage3_Dialogue_State_Tracking/week3)

**[5/10] Advanced DST Models**

- 9강 : Advanced DST Models
- Daily Mission
	- span prediction 시도해보기 
	- span Prediction Model Preprocessor

 

**[5/11] Remaining Challenges**

- 10강 : Remaining Challenges in the TOD/DST
- Daily Mission
	- Slot Co-occurrence 조사해보기 (Training set)

 

**[5/12~14] WEEK3 Special Mission**

**[5/12] Special Mission 1 : Notebook 제공 (논의중)**

 

**[5/13] Special Mission 2 : Additional technique**

- GPT-2를 이용한 CounterFactual goal generation
- Slot Substitution으로 Data Augmentation해보기
	- 4명이요 ⇒ 3명이요

**[5/14] Special Mission 3 : 논의중**

 

### [WEEK 4](https://github.com/ydy8989/boostcamp/tree/main/stage3_Dialogue_State_Tracking/week4)

**[5/17~20] WEEK4 Special Mission**

**[5/19] 석가탄신일 휴강**

**심화 실습자료 제공 예정**

추후 업데이트 예정입니다 ! 