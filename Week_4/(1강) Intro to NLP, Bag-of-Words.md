# (1강) Intro to NLP, Bag-of-Words

자연어 처리의 첫 시간으로 NLP에 대해 짧게 소개하고 자연어를 처리하는 가장 간단한 모델 중 하나인 **Bag-of-Words**를 소개합니다. Bag-of-Words는 단어의 표현에 있어서 one-hot-encoding을 이용하며, 단어의 등장 순서를 고려하지 않는 아주 간단한 방법 중 하나입니다. 간단한 모델이지만 많은 자연어 처리 task에서 효과적으로 동작하는 알고리즘 중 하나입니다. 그리고, 이 Bag-of-Words를 이용해 문서를 분류하는 **Naive Bayes Classifier**에 대해서 설명합니다. 이번 강의에서는 **단어를 벡터로 표현하는 방법, 문서를 벡터로 표현하는 방법**에 대해 고민해보면서 강의를 들어주시면 감사하겠습니다.



## Intro to Natural Language Processing(NLP)

### 이번 과정의 목표

- Natural language processing (NLP), which aims at properly understanding and generating
	human languages(NLG), emerges as a crucial application of artificial intelligence, with the
	advancements of deep neural networks.
- This course will cover various deep learning approaches as well as their applications such as
	language modeling, **machine translation, question answering, document classification, and dialog systems**. 



###  학문적 체계

- NLP(주요 학회 : ACL, EMNLP, NAACL)
	- low level parsing
		- 토크나이징, stemming(어미의 변화에 대한 연구, 어근 추출)
	- word and phrase level
		- 개체명 인식(NER) : 단일 단어 혹은 여러 단어로 이루어진 고유명사를 인식하는 작업
		- POS(part-of-speech) tagging : 문장내에서 워드의 품사나 성분이 무엇인지 알아내는 task
		- noun-phrase chunking 
		- dependecy 파싱
		- coreference resolution 
	- Sentence level
		- 감정분석(sentiment analysis)
		- machine translation
	- Multi-sentence and paragraph level
		- Entailment prediction : 두 문장간의 논리적 내포 및 모순관계 추론
		- question answering  : 독해 기반의 질의응답
		- dialog systems : 대화모델, 챗봇임.
		- summarization
- Text mining(KDD, The webconf(formerly, WWW), WSDM, CIKM, IWSM)
	- 빅데이터와 연관된 경우가 많다. 
	- Extract useful information and insights from text and document data
	- Document clustering (e.g., topic modeling)
	- Highly related to computational social science - 트위터나 소셜 미디어를 분석하여 사회현상 등등을 분석하는데 사용된다. 
- Information retrieval - 정보검색 분야 (주요 학회 : SIGIR, WSDM, CIKM, RecSys)
	- Highly related to computational social science
		- 이미 검색 시스템이 고도화 되어 상대적으로 연구가 더딘 분야이다. 
		- 하지만, 추천시스템 분야는 활발히 연구되고 있다. 



### NLP 분야의 트렌드

- 단어를 벡터로 나타내는 테크닉 w2v or glove
- RNN 계열의 모델
- attention module에 기반한 transformer 모델
- 각 NLP task에 맞는 세부적인 모델 설계로 분화됨
- self-supervised 학습된 모델들(bert, gpt-3)은 특정 태스크에서 벗어나 다양한 태스크를 수행하는 모델로 발전하였다. 
- 모델이 커짐으로 인한 자원적 한계로 전이학습이 트렌드가됨



## Bag of Words

### Bag-of-Words Representation

- Step 1. 유니크한 단어를 모아서 vocab을 구축한다.

	-  Example sentences: “John really really loves this movie“, “Jane really likes this song”
	-  Vocabulary: {“John“, “really“, “loves“, “this“, “movie“, “Jane“, “likes“, “song”}

- Step 2. 각 단어를 one-hot 벡터로 표현한다.

	- Vocabulary: {“John“, “really“, “loves“, “this“, “movie“, “Jane“, “likes“, “song”}

		![image](https://user-images.githubusercontent.com/38639633/107896991-49844d80-6f7b-11eb-9c7e-ef232de8ef90.png)

	- 임의의 두 단어쌍의 **유클리디언 distance**는 $\sqrt 2$ 이고, **코사인 유사도**는 모두 0이다 

	- 즉, 단어의 의미와 상관없이 모두 동일하게 설정된다. 

- 문장/문서는 이러한 one-hot 벡터들의 합으로 나타낼 수 있다. 

	![image](https://user-images.githubusercontent.com/38639633/107897153-c6afc280-6f7b-11eb-9631-79ab2cd67ac6.png)



### NaiveBayes Classifier for Document Classification

위와 같이 Bag of Words 벡터로 나타낸 문서를 정해진 카테고리 혹은 클래스로 분류할 수 있는 대표적인 방법인 NaiveBayes Classifier를 알아보자

![image](https://user-images.githubusercontent.com/38639633/107897302-10001200-6f7c-11eb-97a8-eca67706a481.png){:width="80%"}{: .center}

- Bayes’ Rule Applied to Documents and Classes

	- For a document d and a class c

		![image](https://user-images.githubusercontent.com/38639633/107927387-4d839000-6fba-11eb-90be-6443b810a5d9.png)

	- For a document `d`, which consists of a sequence of words `w`, and a class `c`

	- The probability of a document can be represented by multiplying the probability of each word appearing

	- $P(d\vert c)P(c)=P(w_1, w_2,\dots,w_n\vert c)P(c)\rightarrow P(c)\prod_{w_i\in W}P(w_i\vert c)$ 

	- 특정 카테고리 c가 고정되었을 때, 문서 d가 나타날 확률이고, 이는 $w_1$부터 $w_n$까지 동시에 나타날 동시사건으로 볼 수 있다. 각 단어가 등장할 확률이 서로 독립이면, 이를 곱한 형태로 나타낼 수 있다. 

- Example

	![image](https://user-images.githubusercontent.com/38639633/107929544-2084ac80-6fbd-11eb-9ff9-e79e11777934.png)

	이러한 상황에서 Test 데이터의 *Classification task usses transformer*라는 문장의 각 단어에 대한 조건부 확률은 다음과 같다.

	![image](https://user-images.githubusercontent.com/38639633/107929512-1662ae00-6fbd-11eb-9f1d-0712dac98a25.png)

	For a test document $𝑑_5$ = “Classification task uses transformer”

	-  We calculate the conditional probability of the document for each class
	-  We can choose a class that has the highest probability for the document

- $P(C_{cv}\vert d_5)=P(C_{CV})\prod_{w\in W}P(w\vert c_{CV})=\frac{1}{2}\times\frac{1}{10}\times\frac{1}{10}\times\frac{1}{10}\times\frac{1}{10}=0.00005$ 이다.

