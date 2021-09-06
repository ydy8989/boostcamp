# (4강) Passage Retrieval - Sparse Embedding

**강의소개**

4강에서는 단어기반 문서 검색에 대해 배워보겠습니다. 먼저 문서 검색 (Passage retrieval)이란 어떤 문제인지에 대해 알아본 후, 문서 검색을 하는 방법에 대해 알아보겠습니다. 문서 검색을 하기 위해서는 문서를 embedding의 형태로 변환해 줘야 하는데, 이를 passage embedding 이라고 합니다. 이번 강의에서는 passage embedding이 무엇인지 알아보는 동시에, 단어 기반으로 만들어진 passage embedding인 sparse embedding, 그 중에서도 자주 쓰이는 TF-IDF에 대해 알아볼 예정입니다. 

 

## Introduction to Passage Retrieval

### Passage Retrieval

Query에 맞는 문서(Passage)를 찾는 것. 검색 시스템임.

![image](https://user-images.githubusercontent.com/38639633/132171452-2891f079-06e0-4cb7-a8f3-c9e54a0680df.png)



### Passage Retrieval with MRC

- Open-domain Question Answering : 대규모의 문서 중에서 질문에 대한 답을 찾기
	- Passage Retrieval과 MRC를 이어서 2-Stage로 만들 수 있음.
	- retrieval이 되어야만 해당 지문에서 답을 찾으니깐!

![image](https://user-images.githubusercontent.com/38639633/132171545-9c296346-abf4-4081-ac0d-6716c6e54487.png)



### Overview of Passage Retrieval

Query와 Passage를 임베딩한 뒤 유사도로 랭킹을 매기고, 유사도가 가장 높은 Passage를 선택함

![image](https://user-images.githubusercontent.com/38639633/132171675-ee0978f8-2252-4d44-89a4-d804d60a0d8d.png)

- 질문이 들어오면, 임베딩하고, passage도 임베딩을 하게된다. 

	- 여기서 passage는 미리 해둠

	

## Passage Embedding and Sparse Embedding

### Passage Embedding

- Passage Embedding의 벡터 공간.
- 벡터화된 Passage를 이용하여 Passage 간 유사도 등을 알고리즘으로 계산할 수 있음.

![image](https://user-images.githubusercontent.com/38639633/132171844-19f6059a-2525-4472-b2eb-1bea0986524c.png)



### Sparse Embedding 소개

- **Sparse**하다 : 0이아닌 숫자가 굉장이 적게 있는 임베딩을 의미한다. 

	- ex) bag-of-words

		![image](https://user-images.githubusercontent.com/38639633/132171976-e5f0d641-68d6-43e5-9bb4-8997a5566fe1.png)

- BoW를 구성하는 방법 => n-gram

	- unigram(1-gram): it was the best of times => it, was, the, best, of, times
	- bigram(2-gram): it was the best of times => it was, was the, the best, best of, of, times

- Term value를 결정하는 방법

	- Term이 document에 등장하는지(binary)
	- Term이 몇 번 등장하는지(term frequency) 등..(e.g. TF-IDF)

	![image](https://user-images.githubusercontent.com/38639633/132172368-94c0c72d-dce2-4fef-933e-d8b1f4883195.png)

### Sparse Embedding 특징

- Dimension of embedding vector = number of terms

	- 등장하는 단어가 많아질수록 증가
	- N-gram의 n이 커질수록 증가한다. 

	![image](https://user-images.githubusercontent.com/38639633/132172470-e012ab15-beb5-449e-bfca-269c735fdaa4.png)

- Term overlap을 정확하게 잡아내야 할 때 유용하다. 

- 반면, 의미가 비슷하지만 다른 단어인 경우 비교가 불가능하다. 

	![image](https://user-images.githubusercontent.com/38639633/132172546-bd28fbc4-a40c-4720-99aa-59782df68d39.png)

	

## TF-IDF

### TF-IDF (Term Frequency – Inverse Document Frequency) 소개

- Term Frequency(TF) : 단어의 등장빈도
- Inverse Doument Frequeny(IDF) : 단어가 제공하는 정보의 양
- 특정 단어가 한 문서에서 굉장히 많은 빈도로 사용되지만, 전체 문서에서는 등장 빈도가 낮을 때, 해당 단어가 그 해당 문서 내에서는 굉장히 중요한 역할을 하는 단어라고 판단. 이 단어에 더 중요도를 주는 방식

![image](https://user-images.githubusercontent.com/38639633/132172949-80aaf86d-9826-4909-95c4-d515886ecf95.png)



### Term Frequency(TF)

- 해당 문서 내 단어의 등장 빈도

	- Raw count

	- Adjusted for do length : raw count/ words(TF)

	- Other variants : binary, log normalization, etc.

		![image](https://user-images.githubusercontent.com/38639633/132173044-c686aafe-2296-48b6-97ff-8c7071fecb0c.png)



### Inverse Document Frequency (IDF)

- 단어가 제공하는 정보의 양

	
	$$
	IDF(t) = log\frac{N}{DF(t)}
	$$

	- Document Frequency(DF) = Term t가 등장한 document의 개수
	- N : 총 document의 개수

![image](https://user-images.githubusercontent.com/38639633/132173264-26cc1954-0a32-442d-908b-7be37f17ba75.png)



### Combine TF & IDF

$TF-IDF(t, d)$ : TF-IDF for term $t$ in document $d$, 
$$
TF(t,d) \times IDF(t)
$$

1. 'a', 'the' 등의 관사 => Low TF-IDF
	- TF는 높을 수 있지만, IDF가 0에 가까울 것
	- 거의 모든 document에 등장 => $N\approx DF(t)$=> $log(N/DF)\approx 0$​
2. 자주 등장하지 않는 고유 명사 => High TF-IDF
	- IDF가 커지면서 전체적인 TF-IDF 값이 증가 



### TF-IDF 계산 예시

- 데이터 

	![image](https://user-images.githubusercontent.com/38639633/132177890-8615155b-9acd-42ee-b4dd-19e1e61f593d.png)

- TF

	![image](https://user-images.githubusercontent.com/38639633/132177939-4754ec1a-83b7-41e3-bac2-aea95288ddd6.png)

- IDF

	![image](https://user-images.githubusercontent.com/38639633/132177962-86edc7a2-2a93-48fe-8d27-4c7c29947e75.png)

- TF-IDF 계산

	![image](https://user-images.githubusercontent.com/38639633/132178007-5a700831-da3d-4e43-ac1d-0b3d367e157b.png)



- TF-IDF를 이용해 유사도 구해보기 

	- 목표: 계산한 TF-IDF를 가지고 사용자가 물어본 질의에 대해 가장 관련있는 문서를 찾자

		- 사용자가 입력한 질의를 토큰화

		- 기존에 단어 사전에 없는 토큰들은 제외

		- 질의를 하나의 문서로 생각하고, 이에 대한 TF-IDF 계산

		- 질의 TF-IDF 값과 각 문서별 TF-IDF 값을 곱하여 유사도 점수 계산 
			$$
			Score(D,Q) = \sum_{temr\in Q}TFIDF(term,Q) * TFIDF(erm,D)
			$$

		- 가장 높은 점수를 가지는 문서 선택

		![image](https://user-images.githubusercontent.com/38639633/132179862-76b87665-34ef-4ce7-b301-9015a340d7bd.png)



### BM25?

TF-IDF 의 개념을 바탕으로, 문서의 길이까지 고려하여 점수를 매김
- TF 값에 한계를 지정해두어 일정한 범위를 유지하도록 함
- 평균적인 문서의 길이 보다 더 작은 문서에서 단어가 매칭된 경우 그 문서에 대해 가중치를 부여
- 실제 검색엔진, 추천 시스템 등에서 아직까지도 많이 사용되는 알고리즘

![image](https://user-images.githubusercontent.com/38639633/132180064-ad7573fe-320a-402c-9e1b-b88a6bf39596.png)



**Further Reading**

- [Pyserini BM25 MSmarco documnet retrieval 코드](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-doc.md)
- [Sklearn feature extractor](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) ⇒ text feature extractor 부분 참고

