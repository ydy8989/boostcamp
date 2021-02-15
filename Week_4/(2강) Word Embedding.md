# (2강) Word Embedding

**강의 소개**

단어를 벡터로 표현하는 또 다른 방법인 **Word2Vec**과 **GloVe**를 소개합니다.

Word2Vec과 GloVe는 최근까지도 자주 사용되고 있는 word embedding 방법입니다. Word2Vec과 GloVe는 하나의 차원에 단어의 모든 의미를 표현하는 one-hot-encoding과 달리 단어의 distributed representation을 학습하고자 고안된 모델입니다. Word2Vec과 GloVe가 단어를 학습하는 원리를 중심으로 강의를 들어주시면 감사하겠습니다

**Further Reading**

- [Word2Vec, NeurIPS'13](https://arxiv.org/abs/1310.4546)
- [GloVe, EMNLP'14](https://www.aclweb.org/anthology/D14-1162/)

**Further Questions**

- Word2Vec과 GloVe 알고리즘이 가지고 있는 단점은 무엇일까요?

## What is Word Embedding?

- Express a word as a vector
- 'cat' and 'kitty' are similar words, so they have similar vector representations $\rightarrow$ short distance
- 'hamburger' is not similar with 'cat' or 'kitty’, so they have different vector representations $\rightarrow$ far distance
- 기본 아이디어는 비슷한 의미의 단어가 벡터 공간 상에서 가까운 곳에 위치하게끔 부여하는 것에 있다. 

<br>



## Word2Vec

- 인접 단어를 기준으로 문맥의 단어 벡터를 학습한다. 
- **가정 :** 비슷한 맥락의 단어는 의미가 비슷하다.

### Word2Vec의 Idea

한 단어가 주변에 등장하는 단어들을 통해 그 의미를 알 수있다는 가정에 착안하므로, 주어진 학습 데이터를 바탕으로 원하는 단어의 주변 확률 분포를 예측하게 된다. 

- **Distributional Hypothesis:** The meaning of “cat” is captured by the probability distribution $P(\mathbb{w}\vert cat)$

![image](https://user-images.githubusercontent.com/38639633/107930783-c71d7d00-6fbe-11eb-86ed-0267c60939c7.png)

- 위와 같이 'Cat'이라는 단어 주위에 "meow", "Pet" 등의 단어가 높은 확률로 등장한다는 것을 학습한다. 

<br>

### How Word2Vec Algorithm Works

- Sentence : “I study math.”을 생각해보자

- 토크나이징을 통해 Vocabulary: {“I”, “study” “math”}으로 사전을 구축한다.

- 사전의 사이즈만큼의 dimension을 가지는 one-hot-vector로 vocab을 표현한다.

- 이후 Sliding Window라는 기법을 통해 중심단어를 기준으로 앞/뒤 단어를 나타내는 pair를 표현한다.

- vocab 사이즈가 이 경우 3이므로 입출력 노드 수는 3개로 되고, 히든 레이어의 노드 수는 사용자가 지정한다. 사영되는 벡터공간의 dimension과 같은 크기를 지닌다. 

	![image](https://user-images.githubusercontent.com/38639633/107932850-5a57b200-6fc1-11eb-80ad-5eed289b7eb0.png)

- Hidden layer의 차원을 2차원이라고 가정하자.

- Input node to Hidden layer를 표현하는 $\mathbf{W}_1$는 3차원에서 2차원으로 표현되고, 다시 ouput layer로 간는 $\mathbf{W}_2$는 2차원에서 3차원으로 표현되므로 아래와 같이 표현할 수 있다.

	![image](https://user-images.githubusercontent.com/38639633/107933384-07322f00-6fc2-11eb-981e-62c0af483185.png)

- [0, 1, 0]로 표현되는 단어 “study” 가 input, [0, 0, 1]로 표현되는 단어  “math” 가 target vector라고 할 때, 위와 같이 학습할 수 있다. 

<br>

### Property of Word2Vec

- The word vector, or the relationship between vector points in space, represents the relationship between the words.

- The same relationship is represented as the same vectors.

- 벡터간의 연산이 단어간의 유사도를 반영하여 표현되는 것이 특징이다. 

	![image](https://user-images.githubusercontent.com/38639633/107933815-97707400-6fc2-11eb-8dfb-8bea0d6b8633.png)

<br>

### Property of Word2Vec – Intrusion Detection

Word2Vec을 통해서 할 수 있는 또 다른 Task중 하나 : intrusion detection

- 여러 단어들이 주어졌을 때, 나머지 단어와 그 의미가 가장 상이한 단어를 찾는 task
- 단어별로 나머지 단어와의 euclidean distance를 계산하고 평균값을 계산하여 차이를 발견한다. 

<br>

### Application of Word2Vec

NLP task에 Word2Vec을 이용한 다양한 응용이 이뤄지고 있다. 

- Word similarity
- Machine translation
- Part-of-speech (PoS) tagging
- Named entity recognition (NER)
- Sentiment analysis
- Clustering
- Semantic lexicon building

<br>

## Glove

각 입력, 출력 단어 쌍에 대하여, 학습 데이터에서 두 단어가 한 윈도우에서 동시에 몇 번 등장했는지를 사전에 계산하고, 입력워드의 임베딩벡터간의 내적값이 두 단어가 한 윈도우 내에서 몇 번 '동시에' 나타났는가, 그 값에 로그를 취해 fitting 될 수 있도록 학습하는 방식이다. 

- Rather than going through each pair of an input and an output words, it first computes the co-occurrence matrix, to avoid training on identical word pairs repetitively.
- Afterwards, it performs matrix decomposition on this co-occurrent matrix.

$$
J(\theta) = \frac{1}{2}\sum^\mathbf W _{i,j=1}f(P_{ij})(u^T_iv_j-logP_{ij})^2
$$

**Word2Vec**: 특정한 입출력 단어 쌍이 자주 등장했을 때, 이 같은 데이터 아이템이 여러번에 학습 됨으로써 두 워드 임베딩 내적값이 빈번해지면서 커지는 방식이라면

**Glove** : 애초에 동시에 등장하는 단어 쌍이 동시에 등장하는 횟수를 미리 계산하고 이에대한 로그값을 취한 그 값을 직접 해당 두 단어간의 내적값과 얼마나 차이나는지를 loss로 하여 학습한다. 따라서 중복되는 계산을 줄여주는 점에서 빠르고, 적은 데이터에 대해서도 잘 동작하는 특성을 지닌다. 

<br>

### Linear Substructure

Glove에서도 Word2Vec의 Linear Substructure의 결과처럼 벡터에 따른 word간의 차이가 유사하게 적용됨을 볼 수 있다. 

