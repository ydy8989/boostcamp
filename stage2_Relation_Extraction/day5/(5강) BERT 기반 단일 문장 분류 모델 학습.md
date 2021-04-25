# (5강) BERT 기반 단일 문장 분류 모델 학습

**강의 소개**

3강에서 배운 BERT를 가지고 자연어 처리 Task를 해결해 봅니다. 🧐

단일 문장 분류 모델은 주어진 문장에 대하여 특정 라벨을 예측하는 것입니다.

활용 분야로는 리뷰의 긍정/부정 등의 감성 분석, 뉴스의 카테고리 분류, 비속어 판단 모델 등이 있습니다.



## KLUE 데이터셋 소개

### KLUE 데이터셋

한국어 자연어 이해 벤치마크(Korean Language nderstanding Evaluation, KLUE) : 다양한 task가 포함되어있다. KLUE에 소개된 자연어 task를 모두 해결한다면, 살면서 겪을만한 자연어 task를 모두 해결할 수 있다고 이해할 수 있다.

- 단일 문장 분류
	- 문장분류
	- 관계 추출
- 문장 임베딩 벡터의 유사도(e.g. [CLS]) 
	- 문장 유사도 : 두 문장의 유사도(cosine similarity)가 어느정도 비슷한지. 예로 CLS토큰을 통해 유사도를 구하고 이 것을 활용해 챗봇까지 만들어 나아가는 방식이 있다.
- 두 문장 관계 분류 task
	- 자연어 추론 : 주어진 두 문장이 어떤 관계에 있는지를 분류하는 task. 
- 문장 토큰 분류 task
	- 개체명 인식
	- 품사 태깅
	- 질의 응답
- DST task(stage3)
	- 목적형 대화
- 의존 구문 분석 : ~~처음들어봤을 것이다~~



### 의존 구문 분석

단어들 사이의 관계를 분석하는 task를 의미한다. 

![image](https://user-images.githubusercontent.com/38639633/116002150-51273900-a633-11eb-9b3e-05372d113d1e.png)

1. 특징

	1. 지배소: 의미의 중심이 되는 요소
	2. 의존소: 지배소가 갖는 의미를 보완해주는 요소(수식)
	3. 어순과 생략이 자유로운 한국어와 같은 언어에서 주로 연구된다. 

2. 분류 규칙

	1. 지배소는 후위언어이다. 즉, 지배소는 항상 의존소보다 뒤에 위치한다.
	2. 각 의존소의 지배소는 하나이다. 
	3. 교차 의존 구조는 없다.(위 이미지에서 화살표가 교차되는 구조는 없다)

3. 분류 방법

	1. Sequence labeling 방식으로 처리 단계를 나눈다.

	2. 앞 어절에 의존소가 없고 다음 어절이 지배소인 어절을 삭제하며 의존관계를 만든다.

		![image](https://user-images.githubusercontent.com/38639633/116002319-2d182780-a634-11eb-9f1e-3d71de50e65c.png)

	3. 위와 같은 방식으로 의존과 지배 관계를 만들어간다. 

	

**어디에 쓸까??**

복잡한 자연어 형태를 그래프로 구조화해서 표현 가능하다. 각 대상에 대한 정보 추출이 가능!

![image](https://user-images.githubusercontent.com/38639633/116002402-9861f980-a634-11eb-94f6-9a139ad166ea.png)

- 위 예시에서의 단어들은 정보를 추출하기 어렵다. 특징없는 단어들이 많기 때문이다. 
- 구름 그림은 $\rightarrow$ 새털 구름을 그린것이고 $\rightarrow$내가 그린 것
- 즉, 이 작업을 통해 계층적 구조화가 가능해진다. 
	- "나"는 "구름그림"을 그렸다
	- "구름 그림"은 "새털구름"을 그린 것이다. 
- 특정 정보들을 유연하게 추출할 수 있게 된다. 



## 단일 문장 분류 task

### 문장 분류 task

**주어진 문장이 어떤 종류의 범주에 속하는지를 구분하는 task**

1. **감정분석(Sentiment Analysis)**
  1.  문장의 긍정 또는 부정 및 중립 등 성향을 분류하는 프로세스
  2. 문장을 작성한 사람의 느낌, 감정 등을 분석 할 수 있기 때문에 기업에서 모니터링, 고객지원, 또는 댓글에 대한 필터링 등을 자동화하는 작업에 주로 사용
  3. 활용 방안
    1. 혐오 발언 분류 : 댓글, 게임 대화 등 혐오 발언을 분류하여 조치를 취하는 용도로 활용
    2. 기업 모니터링 : 소셜, 리뷰 등 데이터에 대해 기업 이미지, 브랜드 선호도, 제품평가 등 긍정 또는 부정적 요인을 분석

2. **주제 라벨링(Topic Labeling)**
	1. 문장의 내용을 이해하고 적절한에 범주를 분류하는 프로세스
	2. 주제별로 뉴스 기사를 구성하는 등 데이터 구조화와 구성에 용이
	3. 활용 방안
		1. 대용량 문서 분류 : 대용량의 문서를 범주화
		2. VoC(Voice of Customer) : 고객의 피드백을 제품 가격, 개선점, 디자인 등 적절한 주제로 분류하여 데이터를 구조화
3. **언어 감지(Language Detection)**
	1. 문장이 어떤 나라 언어인지를 분류하는 프로세스
	2. 주로 번역기에서 정확한 번역을 위해 입력 문장이 어떤 나라의 언언인지 타겟팅하는 작업이 가능
	3. 활용방안
		1. 번역기 : 번역할 문장에 대해 적절한 언어를 감지함
		2. 데이터 필터링 : 타겟 언어 이외 데이터는 필터링
4. **의도 분류(Intent Classification)**
	1. 문장이 가진 의도를 분류하는 프로세스
	2. 입력 문장이 질문, 불만, 명령 등 다양한 의도를 가질 수 있기 때문에 적절한 피드백을 줄 수 있는 곳으로 라우팅 작업이 가능
	3. 활용방안
		1. 챗봇 : 문장의 의도인 질문, 명령, 거절 등을 분석하고 적절한 답변을 주기 위해 활용



### 문장 분류를 위한 데이터

**Kor_hate**

- 혐오 표현에 대한 데이터
- 특정 개인 또는 집단에 대한 공격적 문장
- 무례, 공격적이거나 비꼬는 문장
- 부정적이지 않은 문장

![image](https://user-images.githubusercontent.com/38639633/116002876-9c8f1680-a636-11eb-9b04-e522196edadc.png)

**Kor_sarcasm : 비꼼에 대한 문장들**

- 비꼬지 않은 표현의 문장
	- 단순한 욕설은 비꼬지 않은 문장에 속하는 것이 특징
- 비꼬는 표현의 문장

![image](https://user-images.githubusercontent.com/38639633/116002874-98fb8f80-a636-11eb-9a02-941d17549999.png)

**Kor_sae : 질문의 유형을 분류**

- 예/아니오로 답변 가능한 질문
- 대안 선택을 묻는 질문
- Wh- 질문(who, what, where, when, why, how)
- 금지 명령
- 요구 명령
- 강한 요구 명령

![image](https://user-images.githubusercontent.com/38639633/116002863-913beb00-a636-11eb-9b60-6136b6b448b1.png)

**Kor_3i4k**

- 단어 또는 문장 조각
- 평서문
- 질문
- 명령문
- 수사적 질문
- 수사적 명령문
- 억양에 의존하는 의도

![image](https://user-images.githubusercontent.com/38639633/116002857-8bdea080-a636-11eb-8b24-482580c22f7e.png)



## 단일 문장 분류 모델 학습

### 모델 구조

Bert의 [CLS] token의 vector를 classification하는 Dense layer를 사용하여 분류한다.

![image](https://user-images.githubusercontent.com/38639633/116002912-ccd6b500-a636-11eb-8d16-8b17fda4f662.png)



### 주요 매개변수

- input_ids : sequence token을 입력 - 토큰의 vocab상의 id(번호)를 의미
- attention_mask : [0, 1]로 구성된 마스크이며 패딩 토큰을 구분한다.
- token_type_ids : [0, 1]로 구성되었으며 입력의 첫 문장과 두번째 문장을 구분한다.
- Position_ids : 각 입력 시퀀스의 임베딩 인덱스
- inputs_embeds : input_ids대신 직접 임베딩 표현을 할당
- labels : loss 계산을 위한 레이블
- Next_sentence_label : 다음 문장 예측 loss 계산을 위한 레이블



### 학습 과정

- Huggingface의 `Trainer`를 기준으로 

	![image](https://user-images.githubusercontent.com/38639633/116003068-7322ba80-a637-11eb-8b11-2477c4a6767e.png)

	의 순서로 학습이 진행된다. 



**Reference**

classfication

1. [https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b](https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b)

2. [https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/](https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)

3. [https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP](https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP)

4. [https://medium.com/@knswamy/sequence-classification-using-pytorch-lightning-with-bert-on-imbd-data-5e9f48baa638](https://medium.com/@knswamy/sequence-classification-using-pytorch-lightning-with-bert-on-imbd-data-5e9f48baa638)

5. [https://mccormickml.com/2019/07/22/BERT-fine-tuning/](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)



