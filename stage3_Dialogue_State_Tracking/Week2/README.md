### WEEK 2 

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