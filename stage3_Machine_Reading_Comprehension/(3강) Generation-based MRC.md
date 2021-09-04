# (3강) Generation-based MRC

**강의소개**

3강에서는 생성기반 기계독해에 대해 배워보겠습니다. 생성기반으로 기계독해를 푼다는 것의 의미를 이해하고, 어떻게 생성기반 기계독해를 풀 수 있을지 알아보겠습니다. 2강에서와 마찬가지로 모델 학습에 필요한 전처리 단계, 생성기반 모델 학습 단계, 그리고 최종적으로 답을 얻어내는 세 단계로 나눠 생성기반 기계독해를 푸는 방법에 대해 배워볼 예정입니다. 

 

## Generation-based MRC

**MRC의 문제 정의**

1. Extraction-based mrc : 지문 내 답의 위치를 예측 => 분류문제 

	![image](https://user-images.githubusercontent.com/38639633/132093235-33eda0fe-b4b9-46d1-aa54-c686ec7e90f8.png)

2. Generation-based mrd : 주어진 지문과 질의를 보고, 답변을 생성 => 생성문제

	![image](https://user-images.githubusercontent.com/38639633/132093243-cae30f2c-b60d-4f9c-a798-ce6832d1ac76.png)



**Generation-based MRC 평가 방법**

동일한 extractive answer datasets >> extraction-based MRC와 동일한 평가 방법을 사용하기도 함

1. EM score
2. F1 score



**Generation-based MRC Overview**

![image](https://user-images.githubusercontent.com/38639633/132093291-2e8586eb-e144-43ce-aaad-b3f9923e92b0.png)



**Generation-based MRC & Extraction-based MRC 비교**

1. MRC 모델 구조

  Seq-to-seq PLM 구조 (generation) vs. PLM + Classifier 구조 (extraction)

2. Loss 계산을 위한 답의 형태 / Prediction의 형태

  Free-form text 형태 (generation) vs. 지문 내 답의 위치 (extraction) ⇒ Extraction-based MRC: F1 계산을 위해 text로의 별도 변환 과정이 필요하다

  ![image](https://user-images.githubusercontent.com/38639633/132093402-0e869a27-79d0-47bf-bf8f-9adf382e735c.png)

  

## Pre-processing

- 사실상 Extraction 기반 방식보다 더 간단하다. 
- Extraction-based MRC의 경우 정답의 위치를 정확하게 특정해야하는 반면에 Generation-based MRC의 경우, 그럴 필요가 없다. 



### 토큰화

- WordPiece tokenizer 방식을 사용
- WordPiece Tokenizer 사용 예시
	- 질문: '미국 군대 내 두번째로 높은 직위는 무엇인가?’,
	- 토큰화된 질문: ['미국', '군대', '내', '두번째', '##로', '높은', '직', '##위는', '무엇인가', '?']
	- 인덱스로 바뀐 질문: [101, 23545, 8910, 14423, 8996, 9102, 48506, 11261, 55600, 9707,
	19855, 11018, 9294, 119137, 12030, 11287, 136, 102]



### 입력 표현 - Special Token

학습시에만 사용되며 단어 자체의 의미는 가지지 않는 특별한 토큰
- **SOS**(Start Of Sentence), **EOS**(End Of Sentence), **CLS** , **SEP** , **PAD**, **UNK** .. 등등

  ⇒ Extraction-based MRC 에선CLS, SEP, PAD토큰을사용

  ⇒ Generation-based MRC 에서도PAD 토큰은 사용됨. CLS, SEP 토큰 또한 사용할 수 있으나, 대신 자연어를 이용 하여 정해진 텍스트 포맷 (format)으로 데이터를 생성

![image](https://user-images.githubusercontent.com/38639633/132094367-dc791211-316d-4e49-b710-a41b267145b6.png)



### **입력 표현 - additional **information

- Attention mask
	- Extraction-based MRC 와 똑같이 어텐션 연산을 수행할 지 결정하는 어텐션 마스크 존재

- Token type ids
	- BERT 와 달리 BART 에서는 입력시퀀스에 대한 구분이 없어 token_type_ids 가 존재하지 않음
	- 따라서 Extraction-based MRC 와 달리 입력에 token_type_ids 가 들어가지 않음

![image](https://user-images.githubusercontent.com/38639633/132094476-30a5c200-b1a1-4bc3-b2fe-bab4603b58cf.png)

> 좀 advanced한 모델들은 token type ids가 없는 경우가 종종 있다. 스페셜 토큰으로 구분이 가능하기 때문



### 출력 표현 - 정답 출력

- Sequence of token ids
	- Extraction-based MRC에선 텍스트를 생성해내는 대신 시작/끝 토큰의 위치를 출력하는 것이 모델의 최종 목표였음
	- Generation-based MRC는 그보다 조금 더 어려운 실제 텍스트를 생성하는 과제를 수행
	- 전체 시퀀스의 각 위치 마다 모델이 아는 모든 단어들 중 하나의 단어를 맞추는 classification 문제

![image](https://user-images.githubusercontent.com/38639633/132094543-8e52df48-286a-4ea9-a1e5-a2b0866c7448.png)

![image](https://user-images.githubusercontent.com/38639633/132094552-01ef3151-21a7-4850-8b6d-3713d053b0fd.png)



## Model

### BART

기계 독해, 번역, 요약, 대화 등 sequence to sequence 문제의 pre-training을 위한 denoising autoencoder 형태의 구조이다. 

![image](https://user-images.githubusercontent.com/38639633/132094605-30c20e58-3b3a-4af4-8593-1f7bfb73ef56.png)

- bert가 단어 두 개를 마스킹하고 해당 위치를 맞추는 방식이라면 bart는 해당 마스킹 부분을 '생성'하는 방식이다. 



### BART Encoder & Decoder

- BART의 인코더는 BERT처럼 bi-directional
- BART의 디코더는 GPT처럼 uni-directional(autoregressive)

![image](https://user-images.githubusercontent.com/38639633/132097521-d99134f0-6ffb-416d-a33f-d1e2790433d8.png)





### Pre-training BART

BART는 텍스트에 노이즈를 주고 원래 텍스트를 복구하는 문제를 푸는 것으로 pretraining한다.

![image](https://user-images.githubusercontent.com/38639633/132097548-28df8da0-3898-4503-b534-12a1f73a9aea.png)



## Post-proessing



### Decoding

디코더에서 이전 스텝에서 나온 출력이 다음 스텝의 입력으로 들어감 (augoregressive) 맨 처음 입력은 문장 시작을 뜻하는 스페셜 토큰

![image](https://user-images.githubusercontent.com/38639633/132097619-b58790de-300b-4d92-895f-44f9bb671d65.png)



### Searching

![image](https://user-images.githubusercontent.com/38639633/132097636-d5b6e06d-a7f0-46dc-98d0-e4afd4571af6.png)

- Greedy search 
	- decision making을 빠른 시간안에 하지만, 처음 고른 단어가 나중가서 안좋은 결과로 이어질 경우가 발생한다. 
- Exhuaustive search 
	- 모든 가능성을 보는 방식. 
	- timestep에 비례하여 exponential하게 계산량이 증가한다.
	- 문장길이, vocab 사이즈가 커져도 사용하기 힘들다
- Beam search
	- exhaustive search랑 유사하지만 top k의 후보군만을 뽑는다.
	- 다시 그 후보군으로부터 branching out을 통해 결론에 도달한다. 





**Further Reading**

- [Introducing BART](https://sshleifer.github.io/blog_v2/jupyter/2020/03/12/bart.html)
- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) ](https://arxiv.org/abs/1910.10683)

    

