# (2강) Extraction-based MRC

**강의소개**

2강에서는 추출기반으로 기계독해를 푸는 방법에 대해 알아보겠습니다. 추출기반으로 기계독해 문제에 접근한다는 것의 의미를 이해하고, 실제 추출기반 기계독해를 어떻게 풀 수 있을지에 대해 배워볼 예정입니다. 학습 전 준비해야할 단계와 모델 학습 단계, 그리고 추출기반으로 얻어낸 답을 원하는 텍스트의 형태로 변형하는 방법에 대해 이번 강의에서 자세히 알아보겠습니다. 

 

**Further Reading**

- [SQuAD 데이터셋 둘러보기](https://rajpurkar.github.io/SQuAD-explorer/)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning](http://jalammar.github.io/illustrated-bert/))
- [Huggingface datasets](https://huggingface.co/datasets)



## Extraction-based MRC

### Extraction-based/MRC의 정의

- 질문의 답변이 항상 주어진 지문 내에 span으로 존재.

![image](https://user-images.githubusercontent.com/38639633/130887548-6c647f78-c5aa-4462-82ae-32e28d6af7b6.png)



### Extraction-based/MRC의 평가방법

**EM**과 **F1**을 Metric으로 사용한다. 

![image](https://user-images.githubusercontent.com/38639633/130887684-5716421b-087d-4385-b9ba-dc48e1cab795.png)

> 지난 정리에서 확인할 수 있다.



### Extraction-based/MRC/Overview

![image](https://user-images.githubusercontent.com/38639633/130892847-dffb3f6e-31a9-4401-b698-c702985e7aa2.png)

1. context와 question 모두 토큰으로 나뉘어 임베딩된다. 
2. 임베딩은 모델로 들어가서 시작점과 끝점을 output으로 출력하게된다. 
3. 조금 더 엄밀히 말하면, 모델이 내뱉은 output vector를 가장 높은 토큰의 위치로 특정하여 예측하게된다. 





## Pre-processing

### 입력 예시

![image](https://user-images.githubusercontent.com/38639633/130893155-de5a935d-21ad-4f98-84fd-59526afad2a3.png)

- 첫번째 질문에 대한 정답 : 주황색
- 두번째 질문에 대한 정답 : 파랑색



### Tokenization

- 텍스트를 작은 단위로 나누는 것

	- 띄어쓰기 기준, 형태소, subword 등 여러 단위 토큰 기준이 사용된다. 
	- 최근엔 OOV 문제를 해결해주고 정보학적으로 이점을 가진 BPE를 주로 사용한다. 
	- 여기서는 Wordpiece tokenizer 방식을 사용한다. 

	

### Special tokens

- question과 context를 구분하여야 함.
- 스페셜 토큰으로 문장이나 토큰을 구분짓는다. 



### attention mask

- 입력 시퀀스 중에서 attention을 연산할 때 무시할 토큰을 표시하기 위함

![image](https://user-images.githubusercontent.com/38639633/130893607-14edea18-76d4-4795-b9e0-8ef026884136.png)



### Token type IDs

- 입력이 2개 이상의 시퀀스일 때(질문 + 지문), 각각에게 ID를 부여하여 모델이 구분할 수 있도록 유도

![image](https://user-images.githubusercontent.com/38639633/130918286-f032430e-e6b7-4c21-9fd7-3cd6198c8d2b.png)



### 출력 형태

- 정답은 문서내 존재하는 연속된 단어 토큰(span)이므로, span의 시작과 끝의 위치를 알면 정답을 맞출 수 있다. 
- extraction-based에서는 답안을 생성하기 보다 시작위치와 끝위치를 예측하도록 학습한다. 즉, token classification 문제로 치환

![image](https://user-images.githubusercontent.com/38639633/130918523-9b8e9ef5-a06f-480e-b0d4-1397525f5b5b.png)

## Fine-tuning

![image](https://user-images.githubusercontent.com/38639633/130924705-304df3da-7f3a-4ab1-8701-3db9603ba1c2.png)

- 토큰의 시작/끝 토큰일 확률을 모두 예측하고, cross-entropy loss를 통해 이를 예측한다. 



## Post-processing

**불가능한 답 제거하기**

- 다음의 경우 candidate list에서 제거한다. 
	- end position이 start position보다 앞에 있는 경우
	- 예측한 위치가 context를 벗어난 경우
	- 미리 설정한 max_answer_length보다 길이가 더 긴 경우



**최적의 답 찾기**

1. Start/end position prediction에서 score (logits)가 가장 높은 N개를 각각 찾는다.
2. 불가능한 start/end 조합을 제거한다.
3. 가능한 조합들을 score의 합이 큰 순서대로 정렬한다.
4. Score가 가장 큰 조합을 최종 예측으로 선정한다.
5. Top-k 가 필요한 경우 차례대로 내보낸다.



## Reference

[https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/): SQuAD explorer

BERT : Pre training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., NAACL 2019)
[http://jalammar.github.io/illustrated-bert/](http://jalammar.github.io/illustrated-bert/) : TheIllustrated BERT, ELMo,  and  co. (How NLP Cracked Transfer Learning)
[https://huggingface.co/datasets ](https://huggingface.co/datasets):  Huggingface  datasets