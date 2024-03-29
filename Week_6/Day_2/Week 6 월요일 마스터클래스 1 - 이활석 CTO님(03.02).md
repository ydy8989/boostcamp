# Week 6 월요일 마스터클래스 1 - 이활석 CTO님(03.02)

## 사전질문

- OCR분야로 오신지 얼마 안되어서 성과를 내셨는데, 짧은 기간동안 학습효과를 올린 특별한 방법이 있으실까요?
	- 이런말 하면 어떻게 생각하실지 모르겠지만 저는 6개월이면 다들 성과는 올릴 수 있다고 생각합니다(…)
		- 물론 주변 환경과 장비가 다 준비 되어있다는 가정 하에!
		- 제가 OCR할 때에는 OCR 분야가 좀 급해서 지원이 풍부했습니다. 그래서 제약도 적고 연구에 집중할 수 있었어요.
	- 성과 내기 위해 전 회사, 전전회사 동료들을 불러모았어서, 이전에 일해본 동료들과 연구하니 진행이 빨랐습니다.
- AI 분야의 신입 엔지니어에게 이것만큼은 꼭 갖췄으면 한다는 역량이 있다면?
	- 당연히 기본 실력은 보겠죠?
		- 모델링/엔지니어링 등 지원 직무 관련 실력
	- 확인하기 어렵겠지만 꼭 알고싶은 것은 러닝 커브
		- 너무 빨리 트렌드가 변하다 보니까, 알고 있는것을 너무 고수하면 안됩니다.
		- 새로운 게 보이면 뛰어들어서 빠르게 익혀내는 역량이 많이 중요한 것 같습니다.
		- 그런데, 이 러닝커브는 보통 '기본기’가 탄탄하면 좋은 경우가 많습니다.
			- 기본기는 최소한의 코딩실력 / 어느 정도의 수학
		- 또, 내가 잘하던 것을 과감하게 버릴 수 있는 능력도 러닝커브의 일부인것 같습니다.
- 제품화에 대한 이야기를 많이 하시는데요, AI가 제품이 되기 위해서는 어떤 것이 고려되어야 할까요?
	- 강의에서 안 다룬 내용에 대한 질문이겠죠?
	- 예시를 들었던 AI 모델링의 고려사항들 이전에도 많은 판단점들이 있습니다.
	- 일단, '이게 기술적으로 가능한 프로젝트인가?'라는 것을 판단해야합니다.
		- 투입 가능한 리소스와 예산, 시간을 고려했을 때.
	- 또, '비즈니스적으로 의미 있는 영역인가?'를 판단해야합니다.
		- AI로 상상의 나래를 펼치는 사람들이 많은데, 현실적으로 무엇을 해결하는게 의미있는지.
	- 이 두개의 교집합을 잘 찾아야 합니다.
	- 이전에는 전혀 관심이 없었다가, 갑자기 경쟁사들이 AI를 많이 쓰더라! -> 갑자기 AI를 도입하는 경우도…
		- 비즈니스의 감각과 기술의 감각을 둘다 어느정도 이해하고, 커뮤니케이션이 되어야 교집합이 잘 찾아집니다.
- 학습 데이터 셋 담당자의 경우에는 개발자가 담당하나요? 따로 이 일만을 담당하는 포지션이 있나요?
	- 아직까지 한국의 AI 개발팀에서는 개발자들이 많이 하고 있습니다.
	- 개발 인력도 많지 않기 때문에, 소규모 팀은 혼자 다 하는경우도 있습니다.
	- 회사의 문화, 철학과도 맞물려있는것같아요.
		- 어떤 회사는 자체 개발은 안하고, PM 역할만 하고(Model Quality Manager) 나머지는 다 외주를 주고.
		- 어떤 회사는 자체 개발을 하고. -> 이 경우 AI 개발자가 다 하는 경우도…
	- 팀이 커지면 좀 더 세분화되겠죠?
- 강의를 보고 model engineering에 관심이 생겼습니다. model engineer를 목표로 한다면 부스트캠프 (특히 p stage)에서 어떤 부분을 중점적으로 공부하는 것이 좋을까요?
	- P 스테이지에서 직접적으로 얽힌 것은 경량화 파트.
	- 이쪽은 아직 강의나 책이 별로 없는 것 같아요.
		- 책이 나온다는 것은 업무나 분야가 많이 알려져있거나 / 표준화가 되거나 / 경험이 쌓여야하는데…
		- 이쪽은 아직 제가 알고있는 한도 내에서는 별로 없네요.
		- 경량화 관련 수업도 부스트캠프에서 국내 최초로 하는걸로 알고 있습니다.
	- 또, 아직은 모델 엔지니어링 쪽 업무가 많이 변하는 경향이 있습니다.
		- 버전업이 아주 빠르게 일어나고 있어서…
		- 초기 PyTorch나 TensorFlow도 버전업이 아주 빨랐습니다.
			- 안정화되기 전까지 사람들이 많이 쓰고 고치고, 버그 리포팅도 하고…
		- 그래서 모델 안정화 툴까지 있는 현실입니다.
	- U스테이지나 P스테이지 수업이 많이 도움이 되실거에요.
- 지금 부스트캠프를 하면서 강의를 충실히 공부한 후에 시간적 여유가 된다면 부캠에서 강의하지 않는 내용중에 모델링외에 당장어떤것을 배우는 것을 추천하시는지 궁금합니다. 또 그러한 것을 배울 때 스스로 공부할 경우 회사에서 직접 일하면서 배우는 것과는 많이 다를텐데 이러한 갭을 채우는 방법이 있을지 궁금합니다.
	- 모델링 외의 개발력을 많이 키우시는것을 추천드립니다.
		- 오해하실 수 있는데, 지금은 모델링만 잘하셔도 수요가 많아요. 모델링이 당연히 1순위입니다.
		- 그러나 조만간(한 3년~10년)을 내다볼 때 경쟁력을 키우려면 그런 개발력을 키우시는게 좋을 것 같아요.
	- 툴 만드는 프론트엔드, 대용량 데이터 처리하는 백엔드 쪽(하둡, 서버 오케스트레이션은 쿠버네티스 등)
	- 당연히 혼자 공부하는거랑 회사에서 하는거랑 다른건 이미 회사에서도 다 알고 있습니다. 갭이 있을 수 밖에 없어요.
		- 그래도 스스로 혼자 공부하고 오시는 분이 좀 더 빠르게 일인분을 할 수 있어서.
	- 갭을 메우는 방법은 그냥 회사에 와서 빨리 fine-tuning하는 수밖에 없을것같습니다.
		- 간접적으로 해보는건… 학생이시라면 인턴? 인턴은 많이 해보시길.
- AI 개발팀 구조에 대해 배웠는데 궁금한 점이 생겼습니다. 개발 팀이 도메인(NLP, CV, 오디오 등)마다 따로 있는 경우도 있나요?
	- 작은 팀은 여러 도메인을 맡기도 하지만, 거의 대부분은 도메인마다 따로 있어요.
		- 5명이 있다면, 한명씩 도메인 전문가…
		- 각자가 모든 분야를 공부하기에는 양이 너무 많아요.
		- 일단 모델링 팀은 다 따로 있습니다.
	- 근데, 요새는 사용하는 모델 구조가 겹치는(비슷해지는) 경우도 꽤 많아서 점점 협업도 많이 해야하는 분위기로 가고 있어요.
		- 그래도 아직까진 다 따로있습니다.
- AI로의 전직 접근 방법에 대해 다뤄주셨는데, 다른 분야의 커리어가 없는 AI 엔지니어 신입은 어떤 직무를 맡을 수 있을까요?
	- 제 경험을 설명드리면 … 이건 제 팀 운영 철학이랑 맞닿아있는데요.
	- 회사 운영을 위해서는 모든 사람이 회사에 기여를 해야한다고 생각합니다.
		- 회사에 기여하기 위해 드리는 일이 있고, 성장을 위해 드리는 일이 있어요.
		- 성장을 위한 일은 기회를 드리는거니까 차치하고, 기여하기 위한 거라면…
			- 기존의 코드리뷰를 많이 시킵니다.
				- 코드의 잘못된 점도 많이 보겠지만, 코드를 익히도록…
			- 그리고 데이터 리뷰도 많이시킵니다.
				- 쌓아놓은 데이터를 여러 관점에서 분석해보고, 특징을 찾아내고 문제점을 보완하도록 파악해달라.
			- Evalution, 그 중에서도 정성평가
				- 모델 prediction 결과를 보다 보면 전체 업무에 대한 감이 생깁니다.
		- 이것만 해도 몇개월은 가는 거 같아요.
		- 이렇게 하다보면 본인이 잘하는 것 / 기여하고 싶은것이 생기는 경우가 많습니다.
		- 즉, 전체 업무를 보고, 원하는 커리어를 선택할 수 있도록 하려고 합니다.
- 실무에서 데이터분석을 할 때 구축된 데이터셋의 크기가 커서 모든 데이터를 고려하지 않고 샘플링을 할 것 같은데 적절한 샘플링 크기를 어떻게 정하는지 궁금합니다. 데이터의 크기, 성질 따라 샘플링 사이즈도 다르다면 수학적으로 유의미하다고 통계 내릴 수 있는 사이즈가 있나요?
	- 데이터 고려해서 신중하게 샘플링합니다.
		- 예산과 시간이 정해져있으니까.
	- 기술적 의견과 해당 서비스의 도메인 지식이 많이 필요합니다.
	- 샘플링 크기는, 아직까지는 데이터가 많을수록 좋은 경우가 불문율이기에…
		- 한정된 리소스 내에서 max를 고릅니다.
	- 그 이후 조금씩 수정을 하거나 하는 것 같아요.
	- 수학적으로 유의미한 기준…?
		- 그런건 아직까지 없는것같아요.
		- 아직은 대부분 경험이에요.
		- 유사한 task가 논문에 있으면 논문의 데이터를 많이 참조하죠.
		- 아직까진 trial&error의 느낌이 강해요.
	- 데이터 많고 예산 리소스 지원해주고 맘대로 해보세요 라고 하는 프로젝트가 있다?
		- 그럴 땐 이전 경험이 중요하죠.
		- 이전에 어느 사이즈로 딜리버했다.
			- 커브를 보면 대충 어떻게 바뀔지 감이오니까.
		- 분석을 해서 사후평가를 쌓아놨다가, 유사한 task가 오면 그때 적용해보죠.
- 주변 역량을 키우라는 조언을 해주셨는데, 현재 현업에서 ML 엔지니어들이 FE나 BE능력을 가진 비중이 얼마나 되는지 궁금합니다.
	- 거의 없습니다.
		- 거의 없어서 제가 키우면 큰 강점이 될거라고 말씀드리는 겁니다.
	- 모델링하기도 사실 바쁜거같아요…ㅋㅋㅋㅋ
		- 모델링만 하기에 바쁘다는게 아니라, 아직은 모델러가 온갖걸 다 하기때문에…
	- 모델링도 조금, 백엔드도 조금, 프론트엔드도 조금… 이런 분들은 좀 있으신데 이건 좋은 경우인지 모르겠어요.
		- 본인만의 강점을 가질 정도로는 있어야할거같아요.
	- (질문)그럼 신입 입장에서는 어떤 분야를 갈지 탐색하는 시간도 필요하겠네요?
		- 네 그렇긴한데…
		- 일단은 지금 모델링 할 줄 아는 사람도 별로 없어요 ㅋㅋㅋ
		- 길게 봐서는 백엔드/프론트엔드 역량도 좋을것같다는 말이에요.
		- 지금은 백/프론트 역량이 필요하면 그냥 외부 팀이랑 협업하죠.
- 검증된 외부 모델을 사용할 때의 구체적 과정을 듣고싶습니다(외부 코드를 사용하거나 개조할 때 저작권이나 상업적 사용 가능여부, 제작자에게 알려야 하는지 여부 등…)
	- 당연히 회사에서는 라이센스 엄청 중요하구요(특강의 라이센스 강의 잘 들어보세요)
		- 라이센스 꼭 확인하세요.
	- 보통 모델 라이센스 몇개 없거든요? 본인 목적에 따라 찾아보셔서 고르시면 됩니다.
	- 해당된 깃허브 모델이 있다? -> 라이센스 확인하고 -> 우리가 가지고 있는 데이터 태워보고 -> 학습에 공수가 얼마나 들지도 확인하고 (-> 쓸만한 모델인 지 검증하고)
	- 근데 깃허브에 올라온 모델은 학습 코드가 없는 경우가 왕왕 있어요 ㅋㅋㅋ 그건 잘 확인하고 쓰셔야 할 것 같아요.
- 주로 pytorch에서 tensorflow로 변환한다고 하셨는데, 처음부터 Tensorflow로 작성하는 경우는 많이 없는지.
	- 예시를 드렸던 이유가 보통은 pytorch를 많이 하셔서!
	- 저희 팀 같은경우는 tensorflow, pytorch 반반 있어요.
	- 요새는 pytorch->tensorflow 변환도 많이 쉬워져서, 지금은 큰 작업이라고 생각하지 않습니다.
	- 서비스 나가는 정도로 어느정도 모델 구조가 안정화되면, 스크립트로 자동으로 변환되게끔 파이프라인이 짜져 있습니다.
- AI 트렌드를 빨리 캐치할 수 있는 마스터님의 노하우는 무엇일까요?
	- 관련 커뮤니티를 많이 보세요.
		- 텐서플로우 코리아/파이토치 코리아
			- 좋은 논문들 많이 리뷰해서 올라와요.
		- 더 빨리 알고 싶다면 트위터를 하세요.
			- 몇명만 팔로우 해놓으면 빠르게 소식을 볼 수 있어요.
		- 요새는 좋은 뉴스레터도 많아요.
			- [papers with code](https://paperswithcode.com/) - 논문 소식, 주간 소식, 코드 레포지토리 등등…
		- reddit을 쓰시는 분도 계세요.
	- 근데 이건 본인 숙련도에 따라서 다른거같아요. 본인 레벨에 따라 골라서 보세요.
	- 유튜버들도 follow하시면 좋아요.
		- 외국의 유튜버들 - 논문 5분 요약 하시는 등등…
- 마스터님께서 현업에서 서비스향 AI 모델을 개발하면서 직면했던 가장 어려웠던 문제나, 자주 직면한 문제가 있었다면 어떤 문제였는지 그리고 해결한 과정들이 궁금합니다.
	- 맨 땅에 헤딩이 제일 힘들었죠 ㅎㅎ
		- 서비스향 AI 모델 개발 강의도 3-4년간 구르고 나서 정리한 것.
		- 참조할만한 강의가 있는것도 아니고…
	- 사실 아직까지도 맨땅에 헤딩을 많이 합니다.
		- 구글이나 이런데 가서 물어봐도, 잘하고 계시지만 아직도 맨땅에 헤딩을 많이 하세요…
			- 풀어야 하는 문제에 대한 레퍼런스가 별로 없어요.
	- 저혼자 고민해서 되는 문제는 아닌거 같고, 커뮤니케이션을 아주 많이 해야해요.
		- 팀원들 포함, 협력 팀과도 토론을 많이 했어요.
		- 그리고 이런 시도를 했는데 이런 결과가 나왔다? -> 다른 사람들에게 내부전파를 많이 하려고 노력했습니다.
- 마스터님께서는 게임 회사 그리고 네이버에서 다루었던 AI 분야가 꽤나 다른 분야인 것 같습니다. 이처럼 분야 전환시 적응하는 것에는 어려운 점이 없으셨는지, 그리고 적용 기술에 대하여 어떻게 따라가실 수 있었는지 궁금합니다.
	- 어려움이 없진 않은데, 큰 프레임워크는 그렇게 다르지 않으니까…(AI가 이게 좋은건지도.)
	- 한 분야에서 같은 task를 맡더라도 어려움이 있고(트렌드가 바뀌니까), 다른 task를 간다고 해서 더 어려운거까진 모르겠어요.
	- 기술 분야에 민감하시다면, 길이 보실 것 같아요(본인만의 노하우, 러닝커브)
		- AI를 하시다 보면 본인의 노하우가 생기실거에요.
		- 빠르게 익혀서 레벨업을 하는.
		- 러닝커브가 좋으면 어떻게든 하시는거 같아요.
- 이번 부스트캠프로 개발을 시작했다면 어떤 역량을 위주로 키워나가는 것을 추천하시나요?
	- 어쨌든 부캠이 모델링 위주로 가고있으니까, 한 도메인 잡아서 모델링을 더 깊게 파는 것을 추천 드려요.
		- 어 모델링이 진짜 아닌거같다면… 다른 분야를 찾아야겠죠.
	- 보통은 모델링을 하시면 논문을 읽고 구현하는 기술이 기본이거든요. 그 스킬을 좀 더 팔 것 같아요.
- 만약 학사 출신의 신입도 채용을 한다면 석사 출신에 비해 논문 경험, 연구 경험이 훨씬 적을것으로 예상되는데 학사 출신 신입은 무엇을 보고 채용하시는지 궁금합니다.
	- 앞에서 계속 말씀드렸던 학사로서 어느정도의 역량이 갖춰져있느냐…
	- 제일 좋은건 러닝커브
	- 어떻게 받아들이실 지는 모르겠는데… 석사라고 무조건 논문/연구 경험이 많지는 않아요.
		- 석사에서 논문 구현 / 연구 경험이 많으신 분들은 물론 좋겠죠.
		- 학사신데도 회사 인턴을 많이 하시고, 개인 시간 써서 논문 구현하고 이런 분들도 많이 계세요.
		- 석사라면 학사4년 + 석사 2년을 감안해서 더 기대치가 있을텐데…
- 실제 현업에서 일을 하셨는데 (비전공) 신입에게 기대하는 개발 역량 또는 AI기술 역량은 어느정도인지 궁금합니다.
	- 똑같이 볼 것 같아요.(약간 감안한다고 하더라도)
	- 어찌됐든 오셔서 회사와서 1인분 하셔야하잖아요.
	- 컴퓨터 공학이 전공이라고 생각하시겠죠? 근데 업스테이지에서는 컴퓨터공학 전공이 반밖에 안돼요.
	- 비전공자라면 학부 4년동안 다른 걸 공부하셨단 건데, 어찌됐든 이쪽으로 넘어오시려면 AI 쪽 지식은 편견없이 판단하고있습니다.

## 캠퍼분들께 마지막으로 하시는 말씀

- 제 답변에서도 느끼셨는지 모르겠지만, 질문에서 명쾌하게 답변을 드리기 아직 쉽지 않은 것같아요.
	- 저도 그렇고, 모두들 맨땅에 헤딩하는 느낌이라.
- 혼란스러우시더라도, 혼란함에 너무 휘둘리지 마시고 나름의 방향을 정하셔서 꾸준히 달려가시면 좋겠어요.
	- 틀리실수도 있기 때문에 주변의 피드백도 받구요.
- 러닝커브/회사 오셔서 하실 일들을 체험하기에 P스테이지 좋은 기회일 것 같아요.
	- P스테이지 올인하셔서 스스로 공부도 더 해보시고… 경험해보시길 권장드립니다.