# P stage 4 DKT 1주차 금요일 오피스 아워 - Baseline & Pandas - 조대현 멘토님

고려대학교 인공지능학과 석박사 통합과정

정형데이터를 안겪으신 분들이 많아서 Pandas를 간단하게 다뤄봤습니다.
관련해서 모델 output을 찍어놓은 주피터 노트북도 올려드리겠습니다.(갓)

------
 
## Baseline Code 분석

- 주안점은 크게 2가지로
	- 코드가 어떤 형식으로 흘러가는가
	- 데이터가 어떻게 처리되는가
- 타겟은 정말 1도 모르겠다 하는 분들!
- python `train.py`
	- main(args)
		- PREPROCESS
		- trainer.run(ars)
		- SETUP
		- for EPOCHS
			- train(*args)
			- valid(*args)
 
### Arguments
 
- 기본
	- `-v`, `-var` 2가지 형태로 입력 가능
	- default, type, required(필수인자여부), help(설명글)까지 설정 가능
- argument 설명
	- `model_name` : 코드에서 model 이름을 바꿔서 저장해야 매번 같은 이름으로 저장 안됩니다.
	- `max_seq_len` : 시퀀스데이터 중 일부만 잘라서 사용하도록 설정해놨습니다.
- parse_args에 관한 팁!
	- Jupyter에서 argparse를 사용하려하면 에러가 납니다 ㅠㅠ
	- **이 때, 빈 리스트([])를 argparse에 넣어주면 잘 돌아갑니다!**
	- Argparse 데이터는 내부 속성 호출이 dot으로만 가능
		- 두경우 모두 사용하고 싶다면 `easydict` 모듈 사용
- `os.makedirs` vs `os.mkdir`
	- 전자는 subfolder 한번에 생성 가능, 후자는 불가
	- `exist_ok=True`는 디렉토리가 있으면 그냥 넘어간다!

### 파이프라인

```
1. csv 파일에서 데이터를 가져오고
2. max_seq_len으로 자른 데이터를 가져온다.
    - 이 때, max_seq_len를 채우지 못하는 길이의 시퀀스가 존재할 수 있으므로 시퀀스 데이터 앞쪽에 padding을 넣었음!
    - collate_fn으로..
3. Preprocess,__preprocessing에서 전처리는
    - train에 있고 test에 없는 것은 없다.(혹시 몰라서 체크하라고 알려드림)
    - 범주형 인코딩
```

### 꿀팁

- Single Underscore
	- 언더스코어(_)를 하나만 사용하면 inherit이 자동으로 되지 않는다.
	- 예약어와 겹치는 경우를 피하기 위해 자주 사용한다.
	- 마지막으로 수행된 결과값을 저장하므로, _로 그냥 저장했을 경우 _를 호출하면 가장 마지막 결과만 가져온다.
		- **종종 메모리가 큰 무언가를 실수로 변수에 할당하지 않았다면… 꿀팁 ㅎㅎ**
- Double Underscore
	- 외부에서 속성/메소드 접근 방지할 떄 사용(JAVA의 private이라고 생각)
		- 정확히 말하면 mangling이라고 부르는데, 호출해도 조회 안되도록 변수명을 바꿔버리는것(근데 사실 조회가능한 규칙이 있긴함 ㅎ)
	- 상속 방지 효과도 있음
	- 앞뒤로 붙여 사용하면 magic method라고 불리는 메소드를 할 수 있음.
	- 3개 써도 mangling 해줍니다.
- **github 주소 뒤에 1s 붙이면(github1s) vscode형식으로 나와요…**

### 파이프라인 세부 분석

- `Dataset.__getitem__`

	- max_seq_len보다 긴 데이터는 자르고, 짧은 데이터는 뒀다가 나중에 `collate_fn`에서 처리
	- mask 객체 생성

- `Dataset.collate_fn`

	- 앞의 결과는 mask만 max_seq_len을 가지고, 나머지는 mask_seq_len보다 짧지만 패딩을 넣어주진 않았음.
	- collate_fn은 보통 Dataloader 인스턴스 생성시에 argument로 넣어줌.
		- Dataset에서 나오는 반환형테는 customize하기 나름!
		- 그치만 매번 처리가 어려워서 에러가 많이 납니다… 화이팅. 저도 파이썬한테 매번 많이 혼나요.
	- 이 함수를 쓰면 batch로 내보낼때 처리형태를 바꿔줄 수 있음.
	- collate_fn을 처리하면 뒤쪽 결과들도 padding을 넣어주게 됨.

- `DataLoader`

	- num_workers에서 데이터를 불러올 subprocess 여러개 사용가능
		- GPU 작업을 하더라도, 모델이 데이터 불러오는 속도보다 빠르면 GPU가 아무리 빨라봤자 연산에 소용이 없음.
		- 그래서 데이터 불러올 때 CPU를 더 사용해서 빠르게 데이터 전송해주는 역할
		- 많다고 무조건 좋은건 아니니 reference 참고
			- (전재열 캠퍼님) 보통 gpu 개수 * 4정도 사용한다고 합니다.
		- 윈도우즈 10에서는 1보다 크게 설정하면 높은 확률로 에러
	- pin_memory.to(device)를 해서 반환
		- 이렇게 안하면 CPU로 데이터를 받고 to(cuda)를 통해 GPU로 넘기게 된다. 이러면 연산속도가 저하.
		- 그러니까 처음부터 GPU로 받는 역할!
		- **지금 베이스라인 코드에 false로 되어있습니다 ㅠㅠ true로 바꿔주세요…(캠퍼님들 제보)**

- `process_batch` at `trainer.py`

	- '0’이라는 수를 padding으로 인식하기 위해 기존에 사용하는 index를 전부 1씩 shifting해줌.
		- n번째 문항 인코딩 시에 인덱스0부터 들어가는데, 우리는 0을 패딩으로 써야 하므로 기존 인덱스를 바꿔서 겹치는 경우 방지
	- 전체에 1을 더한 후 mask를 곱하면 sequence가 없는 경우 0으로 바뀌어서 자연스럽게 padding으로 사용가능(개이득)
	- gather_index는 기존 사용하던 모델이 있는데, 쓰지않게 되어서 지금은 dummy 변수.
	- 오늘 발견한 문제…
		- (이건 슬라이드 도식과 코드를 직접 보시기 바랍니다! 글로는 설명이 어렵네요.)
		- max_seq_len보다 짧은 경우에 문제가 되는 코드인데, 사실 그런 데이터가 거의 없어서 큰 문제는 없습니다.
		- padding을 뒤쪽에다 걸었으면 문제가 없는데 padding을 앞쪽으로 설정해놔서 문제가 생기네요.
		- 그래서 일단 코드를 수정해서 올려드릴게요.
			- 이거 수정했을때 성능이 0.0001차이났으니까 아마 큰 문제는 없었을 겁니다(…)

- `Trainer`

	- Setups

		- Model setup
			- 총 3개, LSTM, LSTMATTN, BERT
			- LGBM의 경우 노트북 형태로만 제공됩니다.
			- 혹시 못받으신 분들은 5/25 데일리미션 첨부파일 확인하실것!
		- Optimizer setup
		- Scheduler setup
			- 2가지 제공
			- ReduceLROnPlateau
				- 지정한 metric이 점점 saturation하면 lr을 decay하는 형태
				- 즉 metric이 필요함!
			- get_linear_schedule_with_warmup
				- 초기 LR이 너무 클경우 : 수렴을 못함
				- 너무 작으면 : 학습이 안됨
				- 중간을 잡기 힘들어서 처음에 LR을 늘렸다가 다시 조금씩 줄이면서 LR을 조정함.
				- 분산을 좀 줄여준다고 합니다… 정확할지는 모르니 일단 실험해보세요.

	- Train

		- Train/Valid

			- 보고자 하는 결과물이 마지막 문항의 정답여부이기 때문에 `[:,-1]`

			- loss 계산과 파라미터 업데이트는 자주 사용하므로 따로 함수로 분리

				- 이거 매번 만들어주기 귀찮아서 재사용하면 좋습니다!

			- ```
				clip_grad_norm_
				```

				은 parameter가 update 되는 크기를 조절

				- 업데이트 되는 값이 너무 크면 optimal을 찾기 힘드므로 clip을 걸어주는 형태.(이렇게 하면 안했을 때보다 더 조금씩 조정 됨)
				- 이거 빼면 수렴못하고 발산하는 경우가 있으므로 한번 테스트해보세요.

- LSTM

	- ```
		__init__
		```

		- Embedding + Projection

			- 3으로 나누는것은 큰 의미는 없고 그냥 숫자 줄여주기 위하여…

			- ```
				nn.Embedding
				```

				- 들어가있는 정수를 토대로 LOOKUP 테이블을 만들어서 하나의 scalar를 1차원 데이터로 만들어줌.
				- 파라미터
					- 10 : number of embedding - 0부터 9까지의 인덱스만 넣어주겠다.
					- 3 : 차원수 - 3차원으로 확장
				- continuous한 데이터를 embedding으로 넣을 수 없으므로 통과하는 입력은 반드시 index여야 한다!!(=0이상 양의 정수만 가능) - 이거 4강인가에서 누가 질문하셨더라구요.
				- 외적 또는 reshape도 가능할 것 같아요(정확히 못들었습니다)

			- +1을 넣어주는건 padding 포함하는 것.

			- ```
				comb_proj
				```

				- 4개 feature를 각각 embedding하고 concat한 다음 nn.Linear로 더 작은 차원 벡터로 축소
					- 축소시키는 이유는 hidden dimension이 너무 커지면 모델에 부하가 걸리니까…
					- 저희 데이터가 크지않아서 체감이 좀 덜할거같긴합니다.

	- ```
		nn.LSTM
		```

		- 최초의 cell state와 hidden state를 넣어줘야합니다.

		- input_size : 입력값 임베딩 차원

		- hidden_size : hidden state/ cell state 임베딩 차원

		- num_layers : LSTM 층수

		- 지금 dropout은 안 들어가있습니다.(0)

		- ```
			Forward
			```

			- 입력을 넣어주면 최종 출려과 hidden/cell state를 모두 반환
			- 초기 입력은 일반적으로 0으로 초기화해서 넣어줌
				- hidden/cell state initial을 안넣어주면 알아서 0으로 초기화됩니다.

	- LSTM Output

		- contagious

			- reshape 등은 데이터를 실제로 변형하는 것이 아니라 

				메모리는 그대로 둔채로 접근 방법만 바꾸기

				- 그래서 연산속도/성능 향상

			- 근데 contagious는 새로운 텐서를 반환!

		- 어차피 마지막 문항을 맞췄는가만 보기 때문에 Hidden state를 FC로 보낼 때 마지막 sequence만 보내도 됨.

			- 근데 (아주 대충) 실험해보니 일단 다 넣는게 더 좋긴 하네요.

- `LSTMATTN`

	- Embedding 부분은 LSTM과 동일하고, 후술한 BERT도 마찬가지
	- ATTENTION은 BERT 인코더 사용
		- 상훈님 대회 솔루션 참고, 허깅페이스 라이브러리 참고
	- seq2seq+ATTN vs, LSTMATTN vs Transformer
		- 그냥 LSTM에 Attention 걸면 그게 transformer 아닌가요…?라는 질문이 있음.
		- 학습하는 방식에는 차이가 있습니다.
		- LSTM은 확실히 멀리 있는 것들은 점점 saturate되고, transformer는 좀 더 멀리있는 것을 연관해서 이해하도록 학습
		- LSTM ATTENTION은 상훈님이 Riiid 대회에서 높은 등수했던 솔루션을 직접 제공해주셔서 거기서 얻은 아이디어입니다.

- `BERT`

	- ```
		forward
		```

		- 허깅페이스 모델은 forwarding한 아웃풋을 잘 살펴봐야합니다.
		- 우리의 경우 2개의 아웃풋인데 그중 last_hidden_state가 필요(그래서 indexing이 encoded_layers[0])

## 질문

- label encoder로 인코딩 할 때, 'unknown’을 추가하는 이유가 뭔지 알 수 있을까요?
	- 저희는 해당사항이 없긴합니다(test에 있는 애들이 train에 없으면 문제)
- lstm layer에 hidden 값을 전달하지 않고 None으로 넣어도 동일하게 동작한다는 말씀이 맞을까요?
	- (못들었습니다 ㅠㅠ)
- 아까 rolling을 하는 이유를 알 수 있을까요?
	- 두번째 sequence를 가지고 정보가 나오면 그건 두번째 문항을 맞췄냐 아니느냐의 문제.
	- 근데 interaction은 앞에 풀었던 문항이 맞았느냐 틀렸느냐이므로, 그걸 사용하려면 rolling을 해야함(정확히 못들었습니다 ㅠㅠ)

------

## Pandas

- 김명수캠퍼님 질문에 대한 대답 - elapse time 찾을때 time stamp가 푸는걸 시작한 시간이 아니라 답안 제출한 시간처럼 되어있음.
	- **이거 shift(-1)로 수정해놨는데 그게 적용 안되었나봐요 ㅠㅠ**
	- edwith page에 올려놓을게요…

### 과일 dataframe

- 과일별로의 평균을 보고 싶을 때 사용하는것이 `group_by`
	- 특정 컬럼 내의 unique한 값들을 기준으로 df를 묶어놓음.
	- 근데 묶어놓기 때문에 그냥 바로 print 찍으면 generic만 나옴!
	- 그래서 거기서 추가로 함수를 사용했을 때 제대로 보여줌.
		- 근데 그 함수가 뭐가 있는지 잘몰라서 처음엔 사용하기 어렵습니다 ㅠㅠ
		- std, mean. shift, get_group(특정 그룹 가져오기), size
- `crosstab`
	- 특정 컬럼의 value값이 등장했냐 아니냐에 대한 함수
	- 예를들어 crosstab( 과일, 원산지 )하면
		- 각 원산지 별로 망고가 있는지 없는지 등을 알수 있음.
- `pivot` : index를 바꾸어서 특정 column을 인덱스로 이에 대한 column value 값을 볼 수 있음
- custom 함수로 aggregration
	- `percentile`처럼 agg() 내에 특정 컬럼에 대해 customize 함수 넣어서 사용할수도 있음.
- 더 자세한건 허태명 마스터님 stage 2 강의 보세요! 정말 자세히 나와있습니다.

------

## 사전 질문

- Data Augmentation : 목요일 데일리 미션에서 Riiid data augmentation 관련 내용(자세한 부분은 못적었습니다. 오피스 아워 영상 보시길!)
	- data sequence 중 버리기 아까운 데이터가 너무 많아서 그걸 활용하려고 했어요.
	- augmentation할때 주의할 점
		- sequence data에서 기존 데이터를 변형하는 식의 augmentation을 수행할 때 shuffle이 없도록 해야함!
		- plausible한지(말이 되는지)
			- DKT 예시라면,augmentaion 된 사람처럼 푸는 사람이 있을 것인가?
		- augmentation을 해도 되냐 안되냐 여부는 사실 가장 확실한 건 **성능이 올랐냐**입니다…
	- 자료도 나중에 올려드릴게요.
- Missing values : timestamp를 이용해서 시간 관련 feature를 만드는 중입니다. 결측값은 무엇으로 채우는게 좋을까요?
	- 정확히 답변드리기는 힘들지만…
		- ‘시간’ 특성상 linear interpolation이 가장 무난하지 않을까 의견을 조심스럽게 드려봅니다.
	- 5월 27일 목요일 daily mission 코드에 현병님이 굉장히 자세하게 결측치 다루는 방법을 다루어놓아쓰니 참고하시면 좋을 것 같습니다.
		- 현병님 영혼을 갈아넣으셨다고 하니 꼭 보시길…
- BERT Output : BERT 모델 아웃풋의 마지막 시퀀스만 사용한다고 알고 있는데, 아웃풋의 나머지 시퀀스들은 사실상 계산할 필요는 없는 것인가요? 사용하는 모델 구조상 어쩔수 없이 받아오는 것인지 최종결과에 영향을 미치는지 궁금합니다.
	- 앞쪽 BERT 설명에서 정확히 이부분 설명중입니다.
	- 더 넣으시려면 지금 모델 구조상으로는 힘들고 attention을 더 low level로 사용하셔야 할거에요.
	- 다음주 월요일에 Riid Winner 솔루션 코드에 **실제 마지막 시퀀스만 연산에 넣어 연산량을 줄이 사례가 있습니다(스포일러)**
	- 저희도 실험해보고 있는데…
		- Loss 계산하는 부분
		- 입력값 feedforward 하는 부분에서 마지막 sequence만 넣거나 일부를 넣거나 하는 등…
		- 해보시면 좋을것같습니다. 일단 실험은 꼭 해보세요!
- 강의중에 CV 전략이 중요하다고 하셨습니다. 현재 대회에서 어떤 방법을 사용할 수 있을까요?
	- K-Fold 방법을 사용하면 좋을 것 같고, 마지막 sequence의 정답 여부를 바탕으로 K-fold를 진행하실 수 있을것같습니다.
	- DKT 데이터가 엄밀히 시계열은 아니지만… 일단은 가장 마지막 시퀀스를 맞춰야하는 문제니까.
	- (관련 링크도 첨부해주셨습니다)
- Embedding : 같은 모델이라도 Dataframe의 feature가 다를떄마다 임베딩 대상을 서로 다르게 해줘야 성능이 보장되는지 알고싶습니다.
	- 다른 feature는 다른 nn.embedding을 사용해야 하느냐는 질문인가요?
		- 그렇게 생각한다면 **네, 맞습니다!**
- lstm attn에서는 bert encoder만 쓰고, bert에서는 bert model을 사용하셨는데, 두 개 차이가 pooling만 있는 것 같고, 사실상 저희는 pooling된 아웃풋인 0번째는 안 써서 아예 차이가 없을 것 같은데 두 개를 다르게 사용하신 이유가 있으신가요?
	- 차이가 그렇게 많지는 않지만, 있다면
		- 내부에서 돌아가는 embedding이 BERT는 한번 더 들어갑니다.
	- pooling은 사실 안써서 차이가 없지만, embedding이 한번 더 들어가는 점이 차이가 있을거에요.
	- 상훈님한테 찾아가서 여쭤봤는데, hugging face에서 나온 encoder가 bert 말고도 다른것도 많잖아요.
		- 근데 각각 자기 encoder가 다 따로 있어서 encdoer를 앙상블 하셨었대요.
		- 근데 효과가 별로 없어서 encoder 다 삭제하고 BERT 인코더만 남겨두셨다고 합니다.
- DKT도 다른 task들 처럼 pre-trained 모델을 쓰는 경우가 있나요?
	- 아니오 없어요… 유명한 task가 아니고 독보적인 Riiid가 이어서…
	- 뤼이드 테크블로그 들어가면 좋은 글 많은데 저희 데이터가 조금 거기 데이터에 비하면 부족한 면이 있긴 합니다. 그래도 읽어보시면 좋을거에요.

------

## 여담

- 아무리 높아도 0.81(AUC)를 못넘을거라고 예상중…
	- 원래는 0.8도 힘들거라고 생각했는데 벌써 0.8에 근접…

- P stage 4 DKT 1주차 금요일 오피스 아워 - Baseline & Pandas - 조대현 멘토님
	- [Baseline Code 분석](https://hackmd.io/M8bfMd05SkyqDnGinbBxlg?view#Baseline-Code-분석)
	- [질문](https://hackmd.io/M8bfMd05SkyqDnGinbBxlg?view#질문)
	- [Pandas](https://hackmd.io/M8bfMd05SkyqDnGinbBxlg?view#Pandas)
	- [사전 질문](https://hackmd.io/M8bfMd05SkyqDnGinbBxlg?view#사전-질문)
	- [여담](https://hackmd.io/M8bfMd05SkyqDnGinbBxlg?view#여담)

[Expand all](https://hackmd.io/M8bfMd05SkyqDnGinbBxlg?view#)[Back to top](https://hackmd.io/M8bfMd05SkyqDnGinbBxlg?view#)[Go to bottom](https://hackmd.io/M8bfMd05SkyqDnGinbBxlg?view#)

Select a repo

Subscribe