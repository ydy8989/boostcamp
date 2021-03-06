> 부스트캠프 AI Tech 김성익님이 올려주신 내용입니다. 

## 서론

너무 많은 내용들을 강의에 넣으면서 시간이 부족해서 설명을 군데군데 못한 것도 있는데, 어려웠다면 미안합니다…^^

## 사전 질문 답변

- AI 관련해서 추천해주실 만한 책이 있나요?

	- 굉장히 많은데, 아마 목적은 ‘수학공부를 하기 위해서 볼만한’ 책을 물으시는 것같다.

		- 그에 한정해서 답변드리면,

			- 영어를 읽는 데에 문제가 없으시면, 

				‘dive in to deeplearning’

				- 현대적인 내용도 많이 다르고 있고, 수식과 함께 코딩에 관해서도 다루고 있음
				- 다른 책들은 비교적 코딩에 대해서는 부족한 부분이 많음
				- 수식보다 코딩이 더 익숙하다! 하시는 분들께 좋음
				- 파이토치 중심
				- 한글 번역이 조금 되어있는 부분을 읽어보시는것도 추천

			- 한글책을 더 선호하신다면, [‘밑바닥부터 시작하는 딥러닝’](http://www.yes24.com/Product/Goods/34970929)

- AI를 공부하면서 수학의 중요성을 느낍니다. 확률론, 통계학, 선형대수 공부를 어떤식으로 공부하는 것이 좋을까요?

	- 사실 수학 공부하고 나서 코딩 공부하면 시간이 너무 오래걸립니다.
	- 수학과 코딩을 ‘같이’ 공부하는걸 추천합니다.
	- 수식과 코딩을 같이 공부하는 습관을 들이시면, 영어사전 펼쳐놓고 하는 느낌입니다(…)
	- 수학 자체가 인공지능에서 필요한 게 아니라, 수식을 인공지능으로 구현하는게 중요하므로, 연구자가 아닌 ML 엔지니어를 목적으로 하고 계시다면 코딩으로 구현하는게 중요합니다.
	- 위에서 설명한 dive in to deeplearning의 appendix에는 필요한 수학항목들이 쫙 나와있습니다.
		- 영어사전 찾아보듯이 이것저것 찾아보면서 공부하는걸 추천합니다.

- 강화학습 같은 경우 목적함수 설정이 어렵다고 알고 있습니다. 간단한 테스크가 아닌 복잡한 태스크의 목적함수 수학자를 통해서 나오는건지 거듭된 수정을 통해 나오는건지 궁금합니다.

	- reward function을 어떻게 설정할 지는, 학습을 관찰하면서 디버깅하는것이 더 중요하다.
	- 이는 수학자들이 만드는것보다는, 엔지니어들이 고민하다가 만들어내는 경우가 많음.
	- reward function 자체를 학습하는 알고리즘을 사용하는 경우도 있음.
	- 어쨌든, hardcore 엔지니어링이 주가 되는 경우가 많다.
	- 그렇다고 수학자가 필요없다는 건 아니고, 원하는 목적의 결과를 끌어낼 때 수학적으로 효율적으로 접근할 수 있다는 게 중요한것.

- 강의에서 배운 수학뿐만 아니라 AI에는 많은 수학이 쓰이는 것으로 알고있습니다. 이 모델들의 수학적 원리를 다 이해하고 있어야하는 건지, 그리고 이러한 수학적 수식이 실제로 일을 할 때 어떻게 쓰이는지 궁금합니다.

	- 이 모델들의 수학적 원리를 다 이해하고 있어야하는 건지
		- 다 이해하면 좋겠지만, 다 이해하고 일하시는 분은 많지 않습니다.
		- 일류 엔지니어 분들도 그때그때 공부하시고, 그때그때 이해하시고 넘어가십니다.
			- 잘 아는 분들도… 개인적으로 교수님께 질문해오시는 분들도 많습니다.
			- 아주 유명한 전문가분들도 컨설팅을 요청하시곤 합니다.
		- 교수님조차도 필요한건 그때그때 공부하시곤 합니다.
		- 물론, 기초를 모르는 건 아니고, 기초를 알아야 어디서 출발할 지 알 수 있습니다.
		- 이번 주에 배운 내용은 그런 기초적인 내용들입니다.
		- 마치, 영어사전을 찾아보기 위해서 알파벳은 떼야하는 것(…)
	- 이러한 수학적 수식이 실제로 일을 할 때 어떻게 쓰이는지 궁금합니다.
		- 코딩할 때 수식부터 시작을 많이 합니다.
		- 거대한 모델의 특정 파트를 가져와서 쓰고 싶다면, 논문에 있는 수식과 코드가 매칭되는 것을 파악하셔야합니다.
		- 수학적 수식과 논문에서 구현되는 코드들이 어떻게 매칭되는지를 알고 있어야 일할 때 사용할 수 있습니다.

- 추천시스템 및 AI 금융 트레이딩에 관심이 있는데, 알아야할 선행 지식이 있을까요?

	- 추천 시스템은 전통적으로 collaboratic filtering같은 전통적 방식이 있습니다.
		- 선형대수 지식을 필요로 합니다.
	- 요새는 context bandit을 쓰는데, 8강에서 사용하는 basian theory를 기반으로 이론들이 만들어져있습니다.
		- 이런 부분을 알아두시면 도움이 됩니다.
	- 금융트레이딩은 어려운 얘기네요. 너무 광범위해서…
		- 어떤 트레이딩을 하냐에 따라 전략이 달라질 수 있는데, 금융 시스템은 stochastic한 기초를 가정하고 만듭니다.
			- 따라서 통계학적 지식은 당연한 베이스입니다.
			- 어떻게 문제를 포착할지, 어떻게 noisy를 필터링할것인지 등등…
		- AI 금융 트레이딩에 초점을 두면…
			- 강화학습과 관련이 있습니다.
				- 이와 관련된 선행지식을 쌓는게 좋을 것같네요.
					- 벨만 방정식을 이해해야합니다.
						- 기본적인 확률론, 선형대수학
				- 알고리즘만 가져다 쓰고싶다면 상관없지만, 모델을 만들어 쓰시고 싶으시다면…

- ML/DL 리서쳐가 아닌 엔지니어가 수학을 어느정도 알아야 할까요?

	- 본인의 업무에 따라 다르겠죠? ㅎㅎ
		- 교수님이 아는 엔지니어들은 수학을 잘하시고, 좋아하십니다.
	- 어느정도 알아야하느냐가 아니라, 계속 모르는걸 채워나가는 traing을 하셔야합니다.
	- 제일 중요한건, 다 아는게 아니라 그때그때 공부할 수 있는 기초체력(수학적 베이스)를 갖추는것이 중요한 것 같네요.
	- 필요한 수학만 공부하면 되는데, 필요한 수학을 공부하기 위해서 질문을 던지고 답을 이해할 수 있는 기초체력이 필요합니다.
		- 이번 강의들은 그런 기초체력 중의 기초입니다(…)
		- 계속 보면서 복습해보시면 도움이 될거에요.

- 강화학습이 현업에서 적용되고 있나요?

	- 적용되고 있는데, 사례를 물어보시는거겠죠?
		- 카카오 얘기는 회사얘기니까 알려드릴수 없고 ㅎㅎ
		- 아는 회사를 알려드리자면…(어디라고는 말씀을 못드립니다)
			- 교통 관련된 시스템 통제
			- 추천 시스템 설계
			- 다 알만한 회사입니다.
			- 내부적으로 쓰는 시스템은 강화학습으로 굉장히 오랫동안 training합니다.
		- single 지표로 하지않고, 굉장히 distribute?된 커다란 모델에서 사용합니다.
		- 로보틱스나 하드웨어를 다룰때에는 강화학습이 사용하기 어렵긴한데, 휴리스틱문제같은것은 강화학습으로 풀 수 있는 경우가 많아 현업에서 쓰기도 합니다.
		- 이 부분은 수학을 제대로 알고 쓴다기보다는 하드코어한 엔지니어링을 위주로 많이 합니다.
			- 이정도만 답변드릴수 있겠네요.

- 교수님이 부스트캠퍼 수강생이라면 어떤 공부를 중점적으로 하실 것 같으신가요?

	- 박사과정을 받고나서 딥러닝을 공부했는데 아쉬웠던 점은…
		- 코드랑 수학이랑 매칭시킬 수 있는 교재가 없었습니다.
		- 그래서 코드따로 수학따로 공부했어야했습니다.
		- 병행할 수 있는 공부를 하시면 좋을것같습니다.
	- 코드에만 집중하시면
		- 깊이가 떨어진다 -> 문제를 해결해야할 포인트에서 다른사람의 아이디어가 사용된다
		- 문제를 풀 때 획기적인 아이디어는 수학적 포인트에서 나오곤 합니다.
	- 리서쳐라면 코드<->수식으로, 엔지니어라면 수식->코드로 옮길 수 있는 능력이 중요합니다.
	- 그런점에서 dive in to deeplearning이 좋은 책인거같다. 라떼는…
		- 교수님 랩실에서도 추천 많이 합니다.
		- 1학년 인턴들에게 가장 먼저 시킵니다.
		- 수식뿐만 아니라 코드까지 다 follow up하도록 시키고 있습니다.
		- 그런식으로 공부하세요!

- 수학이 많이 어렵습니다. 많은 이론을 설명해주시지만 예제나 예시가 부족하고 이해하기 어렵습니다. 강의내용을 자세히 볼 수 있는 책이나 강의가 있나요?

	- 사실 좀 예제가 부족했던것같기는 합니다.
	- 수학 자체를 공부하고 싶으시다면 Pattern Recognition & Machine Learning(PRML)
		- 사실 전공자를 대상으로 하는 책이라 이것도 전공지식이 좀 들어가긴 합니다
	- 책을 하나 집어서 알려드리기 보다는 키워드를 구글링해보는게 좋을 것 같습니다.
		- 강의에서 영어보다는 한글용어를 설명해드리려 했던 이유도, 블로거들이 써놓은 글이 많기 때문입니다.
		- 그런 곳에 쉬운 예제들이 많이 있습니다.
	- 수학을 자세히 설명하는 강의는 영어강의가 몇개 있긴 한데… 어떤 강의가 쉬울지는 잘 모르겠네요.
		- 지금 제가 설명드린것도 최대한 쉽게 설명한건데… 더 쉽게 설명한다면…
		- 아마 모아놓은 강의보다는, 한파트한파트씩 긴 강의를 들어야 할거 같습니다.

## 실시간 Q&A

- 인공지능 관련으로 공부하고 취직하기 위해서는 대학원이 필수일까요? 대학원을 위해 갖추어야할 조건이 무엇일까요? 대학원에 입학하기 위해선 어떤 준비를 해야할까요?

	- 필수?
		- 솔직히 필수는 아니라고 생각합니다.
		- 그치만 도움은 많이 됩니다.
		- 교수님 랩실 사람들은 : 모르는게 정말 많았는데, 질문할 수 있는 사람이 별로 없었고, 대학원에서 그걸 물어볼 수 있었다는게 좋은 포인트라고 말합니다.
		- 모르는 포인트를 혼자 뚫는건 정말 어려운데, 대학원에서 질문을 받으면서 성장하는 것을 추천 드립니다.
	- 입학하기 위해서는?
		- UNIST 기준으로 말씀드리자면, (서류 통과 전제 하에) 수학은 이번 부스트캠프 수업때 배웠던 내용들만 안다면 충분히 합격할 수 있는 수준입니다.
		- 그 외에는 알고리즘, 자료구조 면접 질문들이 많습니다.
		- 교수님도 수학 질문 보다는 다른 것들을 더 많이 질문하십니다.

- 교수님 LAB 선발 기준은?

	- 여기서 말하긴 좀 그렇고… 메일 주세요…ㅎㅎ

- 딥러닝에서 피처엔지니어링(기존 변수를 결합해 파생 변수를 만드는것)이 결과에 도움이 되나요?

	- 어려운 질문이고, 케바케입니다.
	- 그러나 많은 경우에는 직접 feature를 만들어서 학습시키는게 성능에 큰 도움이 되지는 않습니다.
	- 데이터를 좀 더 파고들었을 때 더 효과적이었던것 같습니다.
	- 즉, 지식을 활용했을때보다, 데이터를 잘 선별하고 더 잘 전처리하는것이 도움이 되었던거같긴합니다.
	- 그렇다고 피처엔지니어링이 도움이 안되는가? 그건 아닙니다.
		- 학습을 좀더 빨리 시키고 싶다거나, 이런 변수를 반영하고 싶다라는 니즈
		- 이런것은 현업에서도 많이 해보는 시도입니다.
		- 엔지니어의 의도에 따라서 필요할땐 넣는게 좋습니다.
		- 이 후는 못들었습니다 ㅠㅠ

- 통계학 수업이 쉽지 않은데, 이해 기반이 될 책이 있을까요?

	- 통계학 자체 이론적인 내용은 위에서 말했던 PRML

	- 일본 저자들이 쓴 

		통계학도감

		- 일본 저자들이 쓴게 수식보다도 직관적인 이해가 좀 더 쉽게 설명되어있음

- 인공지능 대학원에 가려면 얼마정도의 수학공부를 해야할까요?

	- 이 수업 정도면 됩니다.

- 시계열 데이터를 다룰때 데이터의 시기(t)에 따라 오버피팅 될 수 있을거같을때, 어떻게 해야 현재 예측을 잘해낼 수 있을까요?

	- 타임 래그에 따라 모델이 오버피팅 된다는 질문인것같은데,
		- RNM같은 경우는 과거 데이터가 오히려 반영이 안되어서,현재 데이터 위주로 반영되는것이 문제인 경우.
		- LSTM같은 기법들은 그 문제를 해결하려고 나온 기법들.
		- 과거데이터에 오버피팅되고 싶지 않게하려면,
		- 용어의 폭탄이네요. 못적었습니다 ㅠㅠ

- 음성/이미지같은 특수한 비정형 도메인이 아닌 가장 일반적인 table 데이터에서 딥러닝은 tree 기반 모델이나 regression 모델에 대해 어떤 장점을 가지나요?

	- 테이블 데이터도 딥러닝을 써서 성능을 낼 수는 있습니다. 잘 쓰면…
	- Tabular라고 해서 딥러닝이 안되는 공식들은 굳이 머릿속에 박으실 필요가 없습니다.
	- 테이블 데이터가 ‘정말’ 크고 넓을때
		- 그래프 모델을 쓰는것보다도 더 도움이 될떄가 있습니다.
	- 테이블데이터가 어떤 성질을 가지고 있느냐에 따라 다릅니다.
	- 물론 교수님도 테이블데이터라면 일반적으로는 트리계열을 먼저 해보곤 합니다. 딥러닝은 시간이 많이 드니까…

- 딥러닝을 확률론적으로 접근하는것과 역전파를 통해 하는것과 다른관점인가요? 제가 생각하기엔 역전파를 하는 점에서 확률과 통계학이 사용되지 않는 것같아서요. 확률과 통계는 다른 접근인가요?

	- 역전파로 gradient vector를 계산하는건 싱글 데이터포인트에서만.
	- 단일 데이터포인트에서는 확률 개념이 들어가지 않지만, 엄청나게 많은 데이터들에서 학습시킬때는…
		- 어떤 데이터로 어떻게 학습을해서 잘 예측하게 할 것인가?
			- 모든 데이터로 학습하면 로컬 미니멈에 빠지거나… 하는 경우가 많기 때문에
			- 미니배치를 잘 뽑아서 업데이트 하는 과정이 중요합니다.
				- 미니배치, SGD 방법 자체가 확률론이 들어가있습니다.
	- 역전파로 계산한 미분값들을 파라미터로 업데이트할때는, '일부데이터’를 사용하는데, 그 '일부 데이터’를 어떻게 뽑을건지
	- 어떻게 optimal한 값으로 가게 할건지는 확률론/통계학이 쓰입니다.
	- 수업시간에 소개하진 않았지만 drop out등의 기법들은 확률론/통계학 등이 다 사용됩니다.
		- 이번 강의는, 공구 박스를 드린것과 같습니다.
		- 여러 공구박스에서 필요한 툴을 그때그때 꺼내서 쓴다고 생각하시면 됩니다.

- 계속 말씀하시는 엔지니어링은 어떤 것을 말씀하시는걸까요?

	- 엔지니어링이 뭐냐? 이건 긴 질문이네요…
	- 교수님같은 리서쳐들이 문제를 풀 때에는, theoritic하게 아름답게 푸는 것을 많이 합니다.
		- 이론으로 해결하려는 것이 리서쳐의 태도입니다.
	- 엔지니어링으로 문제를 해결할 때는 딱 '저 문제’에 가장 알맞은 해답을 찾는 것을 많이 합니다.
		- 휴리스틱하면서도, 이론과 관련 없이 직관에 근거해 문제를 푸는 것을 엔지니어링에서 많이 합니다.
		- 이게 도움이 될 때도 많습니다.
	- 딥러닝은 철저히 컴퓨터의 하드웨어 파워에 의존해 발전했습니다.
		- 메모리 문제, 병목문제, 데이터 버스를 학습시간에 어떻게 개선할수 있을까
			- 이런것들이 엔지니어들이 갖춰야할 지식입니다.
			- 이게 없고 이론만 있으면 현업에서 크게 도움이 되지 않습니다.
	- 내가 동원가능한 모든것을 다 동원해서 문제를 푼다 - 엔지니어링적으로 중요한 점입니다.
	- 리소스 등을 어떻게 관리하느냐 같은 문제들은 강의에서 많이 다루진 않았습니다.
		- 그런데 SGD의 시작은 (교수님이 생각했을때) 이론적 파트보다 엔지니어링상태에서 GPU를 어떻게 활용했느냐 하는 부분들이 아마도 진짜 큰 도움이 되지 않았을까 합니다.
	- 수학뿐만아니라 하드웨어 리소스도 병행해주시면 좋습니다.

- 논문수식 구현하는것도 dive into deeplearning 보고 수식구현할 수 있으면 될까요?

	- 그 책 자체가 교과서에 있는 수식들을 코드로 구현한것입니다.
		- 논문을 보면서 수식으로 어떻게 구현하는가는 내가 아는 지식이 토대가 됩니다
	- dive into deeplearning은 최근 5년간 논문 수식들을 파이썬으로 직접 구현하고 있습니다.
	- 완전 새로운 내용들은 물론 공부를 추가적으로 하셔야하긴 하겠지만, 80%정도는 커버할수있습니다.
	- 가끔 수학이 heavy한 분야가 있습니다. 그런건 좀 힘듭니다.
		- geometric 같은 분야들… 이런건 대학원 오시는게 도움이 될듯합니다.

- 이번 수학강의를 검색하거나 이해할 수 있을정도라면 논문의 수식도 검색하면서 이해할 수 있을까요?

	- 이번 수학강의만 보면 다 알수있어! 보다는, 찾아보면서 검색하는 습관 자체가 충분히 도움이 됩니다.
	- 이게 뭐지? -> 찾아봐야지 -> 아 이런거네 하는 습관을 기르셨으면.
	- 공부를 할 때 필요한 기초적인 내용을 위주로 말씀드렸습니다.

- aws에서 제공하는 ml툴이 완성도있게 나온걸로 알고있습니다. 이런걸로 개발하는것과 직접 개발하는것의 차이가 있나요? ai 엔지니어가 굳이 필요할까요?

	- 이런건… 실제 서비스를 개발하시는 이활석박사님같은 분께 여쭤보면 좋을것같습니다.

- ai.math에서 r.v말고 random process를 쓰는 경우도 있나요? 있다면 쓰는 경우를 알려주세요.

	- 쓰는 경우가 있는데, 메이저하게 쓰이지는 않고 더 수학적 지식이 필요합니다.
		- 과거에 가우시안 프로세스를 다룰때 더많이 사용되었습니다.
	- 이부분은 PRML을 보시는게 좋을것같네요
	- 그래서 강의에서는 크게 다루지 않았습니다.

- 아직 기초도 부족한 상태인데 특정 도메인(경량화, NLP, 이미지, 추천시스템 등)을 잡고 공부하는게 좋을까요?

	- 개인적으로는 기초를 다 공부하고 공부하는것보다는, 뭔가를 만들고싶다는 self-motivation을 가지고 공부하는게 더도움됩니다.
	- 어차피 하나를 공부하면 다른 것을 공부할때도 더 도움이 됩니다.
	- 어떤걸 공부하다가 어떤 기초가 필요하겠구나 -> 그때 차근차근 공부해보시면 더 도움이 될듯합니다.

- 표본분산을 구할 때 N-1로 나눠주는 부분에 대해서는 어느정도까지의 이해가 필요할까요?

	- 통계학에서 배우는 불편추정량에서 출발한것인데, 머신러닝에서 크게 중요한 개념은 아닙니다.
	- 통계학에서 아주 중요한 개념이라 등장한 것입니다.
	- 최대 우도 추정량은 또 N으로 나누기 때문에…
	- 통계추정을 아시면 이해가 바로 될텐데, 크게 어려운 개념은 아닙니다…

- Causal Learning 이 분야는 수학이 얼마나 필요할까요? 대학원 수준의 확률론이 필요할까요?

	- 지금 교수님 랩실에서 하고 있는데, 이 분야는 수학적 백그라운드가 많이 필요합니다.
	- 이 분야를 공부하시려면 수학적 베이스를 많이 가지고 하세요.
	- 대학원수준의 확률론은 제 수업에서는 필요없는데, 연구자를 목표로 하신다면 꼭 필요하십니다.