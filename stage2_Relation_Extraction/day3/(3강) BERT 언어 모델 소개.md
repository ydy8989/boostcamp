# (3강) BERT 언어 모델 소개

**강의 소개**

BERT는 Bidirectional Encoder Representations from Transformers의 약자로 구글이 공개한 인공지능 언어 모델입니다.🤩

BERT는 주어진 Mask에 대하여 양방향으로 가장 적합한 단어를 예측하는 사전 언어 모델입니다.

이번 강의에는 BERT의 내부 구조에 대해 간략하게 알아보고, BERT를 활용하여 해결할 수 있는 다양한 자연어 처리 Task에 대하여 알아봅니다.🤩

 

**실습 코드 링크**

- [0_Huggingface](https://drive.google.com/file/d/1YjfwILfFFQXZM-jpakRFUicMsSPOAhIw/view?usp=sharing)
- [1_BERT_챗봇](https://drive.google.com/file/d/1WzBpwZLzHltkiwUVjpUjK7la2c60S4lI/view?usp=sharing)





## Bert 모델

### BERT 모델 소개 

![image](https://user-images.githubusercontent.com/38639633/114668273-81213300-9d3b-11eb-8474-9e1b98dbc665.png)

- 위와 같은 방식으로 언어 모델은 발전되어 왔다. 
- 그렇다면 bert를 알아보기 전에 autoencoder를 살펴보자



#### bert

![image](https://user-images.githubusercontent.com/38639633/114979717-544d5700-9ec6-11eb-80aa-65ffd0d6e925.png)

- 인코더와 디코더로 구성된 오토인코더는 입력된 이미지를 압축된 형태로 표현하는 것이 목적이다. 
- 일종의 context vector
- 여기서 decoder의 목적은 다시 원본으로 표현하는 것이 목적이다. 
- input을 compressed data로 표현하는 것이 핵심이다. 
- 그렇다면 **bert**는 어떨까?
- 오토인코더와 input, output의 관점에서 봤을 때 차이점은 **masked**의 사용 유무이다. 
- 자연어 모델을 학습할 때, 원본이 아니라 중간 중간에 마스킹이 되어있는 데이터를 넣고 복원하는 작업을 통해 학습이 더 어려워지게 만든다.



#### GPT와 BERT

![image](https://user-images.githubusercontent.com/38639633/114981826-8ca26480-9ec9-11eb-9b23-6934395a8c28.png)

- 위와 같은 차이를 가지고 있다. 
- GPT2는 원본 이미지를 특정한 시퀀스로 자른다. 이후 모델을 통해 잘려진 데이터의 Next를 예측하는 것을 목표로 한다.


 
#### BERT 모델의 구조

![image](https://user-images.githubusercontent.com/38639633/114982192-19e5b900-9eca-11eb-91bb-149f8e85c49b.png)

- 모델의 구조는 위와 같다. 
- sentence1과 sentence2를 `sep`토큰으로 구분된다. 
- bert 내부의 transformer가 all-to-all network로 구성되어있다.
- 그렇기에 `CLS` 토큰의 출력 벡터가 **sen1**과 **sen2**를 포괄하고 있는 어떠한 벡터로 녹아든다고 가정하고 있다. 
- 실제로 `CLS`토큰이 **sen1**과 **sen2**를 잘 표현하기 위해 마지막에 classification layer를 두어 pretraining을 진행하게 된다. 



#### 학습 코퍼스

아래와 같이 많은 양의 코퍼스를 학습에 사용하였다.

- BooksCorpus (800M-words)
- English-Wikipedia-(2,500M-words-without-lists,-tables-and-headers)
- 30,000-token-vocabulary



#### 데이터의 tokenizing

- **WordPiece** tokenizing
	- 하나하나 다 자르고, 빈도수 기반으로 합치면서 토크나이징을 실행하는 방법
- He-likes-playing $\rightarrow$ He-likes-play-##ing
- 입력 문장을 tokenizing하고,-그 token들로 ‘token-sequence’를 만들어 학습에 사용
- 2개의 token-sequence가 학습에 사용



#### Masked language model

![image](https://user-images.githubusercontent.com/38639633/114982982-0ab33b00-9ecb-11eb-91bb-61b4015c4970.png)

- 위와 같이 오리지널 sentence가 주어졌을 때, CLS와 SEP 토큰을 제외한 나머지 토큰 중 15%로 랜덤하게 선택한다. 

![image](https://user-images.githubusercontent.com/38639633/114983084-274f7300-9ecb-11eb-8685-effbe43f0b2e.png)

- 이렇게 선택 된 토큰 중 80%는 Masking에 사용하고
- 10%는 vocab 내의 아무 토큰으로 replace 한다. 
- 나머지 10%는 바꾸지 않는다

![image](https://user-images.githubusercontent.com/38639633/114983397-89a87380-9ecb-11eb-8889-18844bfeea04.png)

- 최종적으로는 위와 같은 input 데이터가 bert에 들어가게 된다. 



#### 다양한 NLP 실험

- **GLUE datasets**
	- ‒MNLI: Multi-Genre Natural Language Inference
		- ‒두문장의관계분류를위한데이터셋
	- ‒QQP: Quora Question Pairs
		- ‒두질문이의미상같은지다른지분류를위한데이터셋
	- ‒QNLI: Question Natural Language Inference
		- ‒질의응답 데이터셋
	- ‒SST-2 : The Stanford Sentiment Treebank
		- ‒영화 리뷰 문장에 관한 감성분석을 위한 데이터셋
	- ‒CoLA : The Corpus of Linguistic Acceptability
		- ‒문법적으로 맞는 문장인지 틀린문장인지 분류를 위한 데이터셋
	- ‒STS-B : The Semantic Textual Similarity Benchmark
		- ‒뉴스 헤드라인과 사람이 만든 paraphrasing1 문장이 의미상 같은 문장인지 비교를 위한 데이터셋
	- ‒MRPC : Microsoft Research Paraphrase Corpus
		- ‒뉴스의내용과사람이만든문장이의미상같은문장인지비교를위한데이터셋
	- ‒RTE : Recognizing Textual Entailment
		- ‒MNLI와 유사하나,  상대적으로 훨씬 적은 학습 데이터셋
	- ‒WNLI : Winograd NLI
		- ‒문장 분류 데이터셋
- **SQuAD v1.1%질의응답 데이터셋**
- **CoNLL 2003%Named-Entity-Recognition-datasets** 
	- 개체명 분류 데이터셋
- **SWAG:-Situations-With-Adversarial-Generations** 
	- ‒현재 문장 다음에 이어질 자연스러운 문장을 선택하기 위한 데이터셋

이러한 데이터셋들이 중요한 이유는 성능 평가의 정확성과 객관성을 보장받을 수 있기 때문이다. 



**NLP 실험**

위의 데이터 셋으로 아래의 task들을 수행할 수 있다. 

![image](https://user-images.githubusercontent.com/38639633/114984346-9a0d1e00-9ecc-11eb-806d-fd110f68519f.png)

- 단일문장
	- bert 모델에 한 개의 문장이 입력 되었을 때 분류하는 task이다

- 두 문장 관계 분류
	- 입력으로 두 문장이 들어가는 경우이다. 
	- 다음 문장을 예측하는 경우, sentence1이 sentence2의 가설이 된다거나, 유사도를 비교할 수 있다
	- 혹은 paraphrase된 것을 detection하는 task에 적용할 수도 있다.
- 문장 토큰 분류 
	- 개체명 인식이 그 예시이다. 
	- output token 각각의 위에 token classifier를 부착함으로써 토큰을 분류하는 작업을 진행
- 기계 독해 정답 분류
	- 두 가지 정보가 주어진다.(sentence1:질문, sentence2 : 해당 질문에 대한 정답이 포함된 문장)
	- sentence2의 수많은 토큰 중에서 start point와 end point의 위치를 잡아내주는 task이다. 

- 이 모든 task가 달라 보이지만, 사실은 크게 다르지 않다. 



### BERT 모델의 응용

#### 감성분석

네이버 영화 리뷰 코퍼스([https://github.com/e9t/nsmc](https://github.com/e9t/nsmc))로 감성분석

학습 : 150000문장 / 평가 : 50000문장 (긍정1, 부정 0)

![image](https://user-images.githubusercontent.com/38639633/114985521-de4cee00-9ecd-11eb-9136-647327c2f865.png)

- 하나의 single sentence를 입력받아 주어진 문장이 긍정인지 부정인지를 분류하는 task이다. 
- input : 하나의 sentence + 각각의 label
- gma


**관계 추출**



