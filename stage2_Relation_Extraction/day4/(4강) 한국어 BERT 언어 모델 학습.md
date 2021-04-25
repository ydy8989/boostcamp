# (4강) 한국어 BERT 언어 모델 학습

**강의 소개**

이번에는 3강에서 소개한 BERT를 직접 학습하는 강의입니다.

다양한 언어 모델들을 활용하고 공유할 수 있는 Huggingface Hub에 대해 소개하고, 직접 본인의 모델을 공유하는 실습을 진행합니다.🤓



## BERT 학습하기

### Bert 모델 학습 단계

1. Tokenizer 만들기
2. 데이터셋 확보
3. Next sentence prediction(NSP)
4. Masking



#### 왜 새로 학습함? 

- 도메인 특화 task의 경우, 해당 도메인만의 학습 데이터만을 사용하는 것이 더 좋은 성능을 발휘한다는 연구결과가 이미 많이 나와있다. 

- ex) 법률 관련 모델을 만들때, 기존의 bert 모델을 fine tuning 하는 것보다 법률 관련 데이터만을 학습하는 것이 더 좋은 성능을 낸다.!!!

	![image](https://user-images.githubusercontent.com/38639633/115997004-7492b980-a61c-11eb-9687-1cab8aa05ea4.png)

- 위 표에서 맨 마지막 PubMedBERT는 생명과학(?), 생리학 분야의 논문을 전처리한 데이터로 만든 관련 모델이다. 

	![image-20210425231910219](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210425231910219.png)

- 위 표에서 좌측의 목록들은 화학, 생명과학 분야의 자연어처리 task를 의미한다. 대부분의 분야에서 각종 bert pretrained model보다 PubMedBERT가 더 좋은 성능을 발휘하는 것을 확인할 수 있다. 



#### 학습을 위한 데이터 만들기

- 이처럼 원하는 도메인에 맞는 데이터로 새로 학습하는 것이 더 좋은 성능을 내기 때문에, 학습을 위한 데이터를 만들 필요가 있다. 
- bert의 기본 모델의 input 형식에 맞는 `input_ids`, `token_type_ids`등을 만들 필요가 있다. 
- 또한, 이렇게 만든 데이터 셋을 `masking`을 통해 어떠한 형식으로 모델에 학습시킬지를 고민해야한다. 

 



**Reference**

- LM training from scratch
	- [https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=5oESe8djApQw](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=5oESe8djApQw ) 
- 나만의 BERT Wordpiece Vocab 만들기
	- [https://monologg.kr/2020/04/27/wordpiece-vocab/](https://monologg.kr/2020/04/27/wordpiece-vocab/)
	- [https://velog.io/@nawnoes/Huggingface-tokenizers%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%9C-Wordpiece-Tokenizer-%EB%A7%8C%EB%93%A4%EA%B8%B0](https://velog.io/@nawnoes/Huggingface-tokenizers를-사용한-Wordpiece-Tokenizer-만들기)
- Extracting training data from large language model
	- [https://www.youtube.com/watch?v=NGoDUEz3tZg](https://www.youtube.com/watch?v=NGoDUEz3tZg)
- BERT 추가 설명
	- [https://jiho-ml.com/weekly-nlp-28/](https://jiho-ml.com/weekly-nlp-28/)

