# (1강) MRC Intro & Python Basics

**강의소개**

P stage 3 기계독해 강의에 오신걸 환영합니다. 첫 강의에서는 기계독해에 대한 소개 기본적인 파이썬 지식들에 관한 강의입니다. 기계독해란 무엇인지, 어떠한 종류가 있는지, 평가는 어떻게 해야할 지에 대해 알아보고, 자연어 처리에 필요한 unicode / tokenization 개념을 배울 예정입니다. 마지막으로 한국어 기계독해 데이터인 KorQuAD를 살펴보며 실제 기계독해 데이터가 어떠한 형태로 이루어 졌는지 배워보겠습니다. 

 

**Further Reading**

- [문자열 type에 관련된 정리글](https://kunststube.net/encoding/)
- [KorQuAD 데이터 소개 슬라이드](https://www.slideshare.net/SeungyoungLim/korquad-introduction)
- [Naver Engineering: KorQuAD 소개 및 MRC 연구 사례 영상](https://tv.naver.com/v/5564630)



## Introduction to MRC

### Machine Reading Comprehension의 개념

- 기계 독해 : 주어진 지문(context)를 이해하고, 주어진 질의(Query / Question)의 답변을 추론하는 문제

- 아래 그림을 보면 검색창에서 질문을 하면 답변을 얻는 것과 약간 상이한데, 이렇게 context로부터 질문에 대한 답을 얻는 모델을 만드는데, 이 때 context를 방대한 자료로 가정하기 때문

	![image](https://user-images.githubusercontent.com/38639633/130466717-fecf0842-7419-43c2-9715-3f82d914dde7.png)



- 하지만 이렇게 방대한 양에서 검색하는 것은 매우 비효율적임
- 따라서 
	- 어떤 지문에서 답을 찾을 수 있는지 sub context를 찾고
	- 그 subcontext에서 답을 찾는 과정으로 진행
- 이는 **Dialog system**에서도 활용되는 방식



### MRC의 종류

**1. Extractive Answer Datasets**

: 질의에 대한 답이 항상 주어진 지문의 segment로 존재. 즉 추출하는 방식으로 두 가지의 세부 task로 정의할 수 있다. 

- 질문에 빈칸을 뚫어놓고 해당 빈칸에 들어갈 entity를 맞추는 방식

	![image](https://user-images.githubusercontent.com/38639633/130467810-6555590b-68fd-4676-9551-03fe097b77ca.png)

- 진짜 질문 형식으로 주어지는 방식

	![image](https://user-images.githubusercontent.com/38639633/130467866-287c16fe-6874-4340-9920-bf000561d990.png)

의 두 가지 방식으로 구성된다. 



**2. Descriptive / Narrative Answer Datasets**

: 답이 지문 내에서 추출한 span이 아니라, 질의를 보고 생성 된 sentence(or free-form)의 형태

![image](https://user-images.githubusercontent.com/38639633/130468191-ce8f8f6d-e69e-4680-8da8-f5160008040c.png)

- 위와 같이 질문에 문장 형식으로 답변할 수 있는 task이다. 



**3. Multiple-choice Datasets**

: 질의에 대한 답을 여러 개의 answer candidates 중 하나로 고르는 형태

![image](https://user-images.githubusercontent.com/38639633/130468804-d58d3bf4-247a-4af5-8b79-651c194a79ba.png)



### Challenges in MRC

**DuoRC(Paraphrased paragraph)**

단어들의 구성이 유사하지는 않지만 동일한 의미의 문장을 이해해야만 한다. (같은 의미인데 다른 단어들로 이루어진 문장을 이해해야하는 문제)

![image](https://user-images.githubusercontent.com/38639633/130737111-9222a1c1-92db-44dc-a499-0ac50a1c33d9.png)

- P1과 P2 둘다 답변으로 될 수 있지만, P2가 더 어렵다. 
- 단순히 토큰을 찾는 문제가 아니라 **문맥적으로 정확히 이해**하는 것이 힘들다. 



**QuoRef(coreference resolution**

대명사를 활용 했을 때, 그 해당 대명사가 무엇을 지칭하는지를 알아야하는 문제

![image](https://user-images.githubusercontent.com/38639633/130737140-c036b5a3-4483-488a-9803-c4dfb9135209.png)



**Unanswerable questions**

지문에서 질문에 대한 답이 없는 경우 

![image](https://user-images.githubusercontent.com/38639633/130737413-00725bd6-7464-41f1-9758-a14f23d2ef4f.png)

- "later laws"와 같이 답을 내놓긴 하지만, 이는 **정확한** 답이라고 할 수 없다. 
- "지문에서 답을 찾을 수 없다"와 같은 답이 더 적절하다. 



**Multi-hop reasoning**

하나의 답을 위해서 지문 속 여러 군데의 답을 취합해야하는 문제. 여러 개의 document에서 질의에 대한 supporting fact를 찾아야지만 답을 찾을 수 있다. 

> HotpotQA, QAngaroo

![image](https://user-images.githubusercontent.com/38639633/130737780-abc16f22-cdf1-49d3-b424-dfdf9c1638a7.png)





### MRC의 평가방법

#### Exact Match / F1 Score

- extractive answer와 multiple-choice answer에 많이 사용된다. 

- Exact match : 

	- 예측한 답과 ground-truth가 정확히 일치하는 샘플의 비율
	- (Number of correct samples) / (Number of whole samples)

- F1 Score : 

	- 예측한 답과 groound-truth 사이의 token overlap을 F1으로 계산

		![image](https://user-images.githubusercontent.com/38639633/130738313-bd6dbec2-a456-4499-a824-2293fa290bdf.png)



#### ROUGE-L / BLEU

- For descriptive answer datasets에 사용 
	- F1을 사용할 수도 있지만, F1의 경우 토큰에 대해서만 채점하기 때문에 답변의 언어적인 부분을 체크하긴 힘들다. 
- ROUGE-L Score
	- 예측한 값과 ground-truth 사이의 overlap recall
	- -L(LCS : Longest common subsequence 기반)
- BLEU
	- 예측한 값과 ground-truth 사이의 Precision
	- n-gram의 겹치는 비율을 계산한다. 

![image](https://user-images.githubusercontent.com/38639633/130738845-d741ad3c-6c15-40a7-85aa-d22cbeacde95.png)



## Unicode & Tokenization

### Unicode란?

전 세계의 모든 문자를 일관되게 표현하고 다룰 수 있도록 만들어진 문자셋 각 문자마다 숫자 하나에 매핑한다. 

![image](https://user-images.githubusercontent.com/38639633/130739054-7b8706aa-6f91-4b93-ba16-6c73fba480bc.png)



### 인코딩 and UTF-8

인코딩이란? 

- 문자를 컴퓨터에서 저장 및 처리할 수 있게 이진수로 바꾸는 것

UTF-8

- UTF-8는 현재 가장 많이 쓰는 인코딩 방식이다. 문자 타입에 따라 다른 길이의 바이트를 할당한다.

- 이유는 메모리를 절약하기 위해서, 문자가 추가될 수록 바이트를 늘리는 방식으로 

	1-byte : Standard ASCII

	2-bytes: Arabic, Hebrew, most European scripts

	3-bytes: BMP(Basic'Multilingual'Plane)' 대부분의 현대 글자 (한글 포함)

	4 bytes: All Unicode characters - 이모지 등



> Python에서의 Unicode는 Python2와 3가 다르니 유의하도록한다. 



### Tokenizing

- 텍스트를 토큰 단위로 나누는 것. 

- 단어, 형태소, subword 등 여러 토큰 기준이 사용된다. 

- subword 토크나이징

	- 자주 쓰이는 글자 조합은 한 단위로 취급하고, 자주 쓰이지 않는 조합은 subword 단위로 쪼갠다. 
	- ##은 디코딩을 할 때 해당 토큰을 앞 토큰에 띄어쓰기 없이 붙인다는 것을 의미한다. 

	![image](https://user-images.githubusercontent.com/38639633/130743944-a165a40d-255d-4dfc-a535-59f7322fafa8.png)

**BPE(바이트 페어 인코딩)**

- 데이터 압축용으로 제안된 알고리즘.
- NLP에서 토크나이징용으로 활발하게 사용되고 있다. 
- 버트에서는 bpe를 변형한? 유사한? wordpiece 방식을 사용하고 있다. 



## Looking into the Dataset - KorQuAD

### KorQuAD란?

LG-CNS가 AI-언어지능 연구를 위해 공개한 질의응답/기계독해 한국어 데이터셋

- 인공지능이 한국어 질문에 대한 답변을 하도록 필요한 학습 데이터셋
- 1,550개의 위키피디아 문서에 대해서 10,649건의 하위 문서들과 크라우드 소싱을 통해 제작한 63,952개의 질의응답 쌍으로 구성되어 있음 (TRAIN 60,407 / DEV 5,774 / TEST 3,898)
- 누구나 데이터를 내려받고, 학습한 모델을 제출하고 공개된 리더보드에 평가를 받을 수 있음
	- 객관적인 기준을 가진 연구 결과 공유가 가능해짐
- 현재 v1.0 & v2.0 공개:2.0은 보다 긴 분량의 문서가 포함되어 있으며, 단순 자연어 문장 뿐 아니라 복잡한 표와 리스트 등을 포함하는 HTML 형태로 표현되어 있어 문서 전체 구조에 대한 이해가 필요



**KorQuAD 데이터 수집**

![image](https://user-images.githubusercontent.com/38639633/130745045-e2079fe1-6c7f-43c6-8aee-dc1f2f40a84b.png)

- SQuAD v1.0의 데이터 수집 방식을 벤치마크하여 표준성을 확보하였다. 



**KorQuAD 예시**

![image](https://user-images.githubusercontent.com/38639633/130745335-93bf5fbb-4ab5-42a7-8338-6499d0f1b0d9.png)

- answer_start가 존재하는 이유는 같은 토큰이 여러개 나왔을 수도 있고, 학습에 혼선을 주지 않기 위해서이다.