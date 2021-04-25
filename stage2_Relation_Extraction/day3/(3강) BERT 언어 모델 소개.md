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

#### Single sentence classification

![image](https://user-images.githubusercontent.com/38639633/115987535-65961200-a5f0-11eb-9ad2-a5c8bf60b0b1.png)

다음의 두 task는 input이 하나의 문장으로 들어가는 단일 문장 분류에 속하는 task이다.



**1. 감성분석**

네이버 영화 리뷰 코퍼스([https://github.com/e9t/nsmc](https://github.com/e9t/nsmc))로 감성분석

학습 : 150000문장 / 평가 : 50000문장 (긍정1, 부정 0)

![image](https://user-images.githubusercontent.com/38639633/114985521-de4cee00-9ecd-11eb-9136-647327c2f865.png)

- 하나의 single sentence를 입력받아 주어진 문장이 긍정인지 부정인지를 분류하는 task이다. 
- input : 하나의 sentence + 각각의 label

	

**2. 관계 추출**

![image](https://user-images.githubusercontent.com/38639633/115987353-a9d4e280-a5ef-11eb-964b-bf149a7270e5.png)

- 문장 내 entity의 관계를 분류하는 task
- single sentence classification인 이유는 두 개의 entity와 sentence 하나가 합쳐서 1개의 문장으로 input되기 때문이다. 



#### Sentence pair classification

![image](https://user-images.githubusercontent.com/38639633/115987614-c4f42200-a5f0-11eb-9988-e58c22d28049.png)

**의미 비교**

디지털 동반자 패러프레이징 질의 문장 데이터를 이용하여 질문-질문 데이터 생성 및 학습
학습:\3,401\문장 쌍 (유사 X:\1,700개,\유사 O:\1,701개)
평가:\1,001\문장 쌍 (유사 X:\500개,\유사 O:\501개)

![image](https://user-images.githubusercontent.com/38639633/115987433-16e87800-a5f0-11eb-946b-d5372ad91968.png)

- 두 문장이 의미적으로 같냐 vs 다르냐를 분류하는 task이다. 
- sentence1과 sentence2 두 개의 문장이 input으로 들어가며, 두 문장의 의미가 같은지 여부를 binary classification하는 문제이다. 
- 여기서 고려해야하는 문제는 데이터 전처리의 문제이다. 
	- `label 0`으로 되어있는 관계없는 문장의 경우, 너무 상관없는  두 문장으로 이루어졌을 때 다른 곳에서 사용할 수 없는 모델이 될 가능성이 높다. 
	- 실제 분류해야하는 문장은 굉장히 의미가 유사하기 때문에, 이렇게 너무도 다른 두 문장을 학습하는 것은 별 도움이 되지 않는다. 



#### Sentence token classification

![image](https://user-images.githubusercontent.com/38639633/115987594-a2620900-a5f0-11eb-81d2-243f35290852.png)

문장의 token을 분류하는 task이다. 



**개체명 분석**

ETRI\개체명 인식 데이터를 활용하여 학습 및 평가 진행 (정보통신단체표준 TTA.KO-10.0852)
학습:\95,787\문장 /\평가:\10,503\문장

![image](https://user-images.githubusercontent.com/38639633/115987637-dfc69680-a5f0-11eb-9e1a-7df285b84af4.png)

- bert 모델을 통과하면서 각 토큰별로 나오는 output과 label을 매칭시켜 학습하는 방식이다. 
- 위의 두 방식(1sentence classificataion과 2sentence classification)은 마지막 layer의 cls 토큰에만 classifier를 붙여 분류하는 방식이었지만, 그림과 같이 모든 토큰에서의 label(개체명)을 분류하는 작업이 이루어지는 task이다. 



#### Machine Reading Comprehension(기계독해)

![image](https://user-images.githubusercontent.com/38639633/115987811-9e82b680-a5f1-11eb-9f12-321bab968ef9.png)

- question과 paragraph를 input으로 갖고 있다. 

- 토크나이저에 따라 성능평가에 많은 영향이 있다. 

	![image](https://user-images.githubusercontent.com/38639633/115992218-79993e00-a607-11eb-85a8-8582d6c0e59d.png)



### 한국어 BERT 모델

#### ETRI KoBert의 tokenizing

- 한국어로 학습된 한국어 모델

- 오랜 기간동안 KorQuAD에서 1등을 유지하고 있던 모델이다.

- wordpiece를 사용하였지만, 음절이 아닌 형태소 단위로 분리를 한 뒤 wordpiece를 태운 토크나이징 방식을 사용하였다. 

	![image](https://user-images.githubusercontent.com/38639633/115992305-ef050e80-a607-11eb-8a17-a5fda4be4420.png)

- 구글의 base bert보다 10점 가량의 성능이 향상되었다. 

- ETRI 형태소 분석기를 사용해야만 사용할 수 있다. 



#### 한국어 토크나이징에 따른 성능비교

> [https://arxiv.org/abs/2010.02534](https://arxiv.org/abs/2010.02534)에서 확인할 수 있다.

- 형태소 분석 후 wordpiece를 태운 것이 성능이 가장 좋았다고 주장하는 논문이다. 



#### Advanced BERT model

>  KBQA에서 가장 중요한 Entity 정보가 기존 BERT에는 포함되어있지 않다. Entity linking을 통한 주요 entity 추출 및 entity tag 부착과 entity embedding layer 추가 그리고 형태소 분석을 통해 NNP와 entity 우선 chunking masking을 통해 KorQuAD에서 좋은 성능을 받을 수 있었다. -김성현 마스터

![image](https://user-images.githubusercontent.com/38639633/115992491-ec56e900-a608-11eb-89d5-6cdd3eb1f66d.png)

![image](https://user-images.githubusercontent.com/38639633/115992498-fbd63200-a608-11eb-9fbf-1ae55a51a00d.png)

- 마지막 줄 Entity Embedding layer가 추가되었다. 
- 그 결과 다음의 성능을 얻었다고 한다.

![image](https://user-images.githubusercontent.com/38639633/115992559-48217200-a609-11eb-807f-15346c2662cf.png)

- 20G의 학습데이터와 달리 더 적은 모델로 적은 학습을 시켰음에도 좋은 성능을 발휘하였다고 한다.
- 언어 모델을 학습할 때 feature가 무엇인지를 고민하고, 사람이 자연어 처리할 때 어떠한 feature를 사용할 것인지 그리고 그것을 모델에 녹인다면 좋은 성능을 발휘할 수 있을 것으로 생각된다. 


```python
!pip install transformer

from transformers import AutoModel, AutoTokenizer, BertTokenizer
# Store the model we want to use
MODEL_NAME = "bert-base-multilingual-cased"

# We need to create the model and tokenizer
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

- `!pip install transformers`를 통해 간단히 설치할 수 있다. 
- 모델 Name에 따른 모델 불러오기와 토크나이저 불러오기가 가능하다. 
- Automodel, AutoTokenizer를 통해 자동으로 모델에 맞는 토크나이저 및 모델을 load할 수 있다. 
	- 기존에는 bert모델을 사용하려면 bertmodel, bertTokenizer를 불러오고, Electra 모델을 사용하려면 ElectraModel, ElcetraTokenizer를 불러와야만 했다 
	- 하지만, 이를 자동으로 매핑해주는 Auto - 시리즈를 통해 이제는 Name만으로 간단히 지정할 수 있다
	- (주의) 그렇지만, 항상 완벽하지는 않다. 특정 모델에는 버그가 존재하여 제대로 불러와지지 않는 현상이 간혹 있다고 한다.

```python
print(tokenizer.vocab_size)
>>> 119547
```
- 11만개의 vocab으로 이뤄진 토크나이저 사전임을 확인할 수 있다. 
- 위 tokenizer는 구글에서 공개한 다국어 모델 학습에 사용된 'bert-base-multilingual-cased' tokenizer 사전이며 약 12만개의 wordpiece 토큰들 중 약 **8천개** 가량만이 한국어이다. 
	- ~~어지간하면 사용하지마라...~~
- 워드피스 기준 vocab을 정의하려면 약 3만개정도로 사이즈를 지정하면 한자어도 인식가능한 정도의 코퍼스를 제작할 수 있다. 

```python
for i, key in enumerate(tokenizer.get_vocab()):
    print(key)
    if i > 20:
        break
```

```
>>>
Vol
Estadual
##երը
সংস্করণ
Voogd
RTL
nghề
##ंगा
##шина
Europese
1001
##هار
##优
##யில்
##lmesi
Sioux
##ლე
##ətli
§
```



간단한 토크나이징 예시를 살펴보면

```python
text = "이순신은 조선 중기의 무신이다."
tokenized_input_text = tokenizer(text, return_tensors="pt")
for key, value in tokenized_input_text.items():
    print("{}:\n\t{}".format(key, value))
```

```
input_ids:
	tensor([[   101,   9638, 119064,  25387,  10892,  59906,   9694,  46874,   9294,
          25387,  11925,    119,    102]])
token_type_ids:
	tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask:
	tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```

- tokenizer에 text를 넣고 pytorch로 반환하겠다는 의미의 'pt' 인자를 넣어주면 객체를 생성한다.

```python
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
input_ids = tokenizer.encode(text)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```
['이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.']
[101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102]
[CLS] 이순신은 조선 중기의 무신이다. [SEP]
```

- `tokenizer.tokenize(text)`를 통해 위와 같이 토크나이징이 된 모습을 직접 확인할 수 있다. 
- `.encode()`는 텍스트를 인코딩 결과를 출력해준다. 
	- 위에서 101번과 102는 문장의 시작과 끝을 알려주는 토크나이저이다. 
	- 이를 `.decode()`를 통해 다시 변환하면 `[cls]`와 `[sep]`로 변환 되어있는 것을 확인할 수 있다. 
- `tokenizer`의 default로 앞 뒤의 스페셜 토큰이 붙는다. 
- 이것을 원치 않는다면 아래와 같이 `add_special_tokens=False`옵션을 통해 넣지 않을 수 있다.

```python
tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
print(tokenized_text)
input_ids = tokenizer.encode(text, add_special_tokens=False)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```
['이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.']
[9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119]
이순신은 조선 중기의 무신이다.
```

---

특정 역할을 위한 special token도 추가할 수 있다.

```python
text = "[ENTITY]이순신은 조선 중기의 무신이다.[/ENTITY]"
# [ENTITY]이순신[/ENTITY]
tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
print(tokenized_text)
input_ids = tokenizer.encode(text, add_special_tokens=False)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```
['[', 'EN', '##TI', '##TY', ']', '이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.', '[', '/', 'EN', '##TI', '##TY', ']']
[164, 38702, 59879, 11517, 166, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 164, 120, 38702, 59879, 11517, 166]
[ ENTITY ] 이순신은 조선 중기의 무신이다. [ / ENTITY ]
```

- 스페셜 토큰을 추가하지 않을 경우, cumstom 토큰은 영문 그대로 인식하여 분리된다. 
	- `ENTITY`토큰이 각각 분리된 것을 볼 수 있다. 
- 아래와 같이 `add_special_toekns`을 통해 스페셜 토큰을 등록할 수 있다. 

```python
text = "[SHKIM]이순신은 조선 중기의 무신이다.[/SHKIM]"

added_token_num += tokenizer.add_special_tokens({"additional_special_tokens":["[ENTITY]", "[/ENTITY]"]})
tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
print(tokenized_text)
input_ids = tokenizer.encode(text, add_special_tokens=False)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
decoded_ids = tokenizer.decode(input_ids,skip_special_tokens=True)
print(decoded_ids)
```

```
['[ENTITY]', '이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.', '[/ENTITY]']
[119552, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 119553]
[ENTITY] 이순신은 조선 중기의 무신이다. [/ENTITY]
이순신은 조선 중기의 무신이다.
```


























































