**Competetion 안내**

**[중요 공지]**

DST 강의를 신청해주셔서 감사합니다:)

이번 Competition은 6강 Open-Vocab DST 실습을 공부하신 후에 도전하시는 것으로 계획하였습니다.

많은 분들에게 생소한 내용인 만큼 무리하지 마시고 차근차근 공부해가시길 바랍니다.

 

1. 대회 개요

	대화 상태 추적(Dialogue State Tracking)은 미리 정의된 시나리오에 안에서, 유저와의 대화에서 특정 정보(Slot)가 의도 된 상태인, 대화 상태 (Dialogue State)를 매 턴마다 추론하는 테스크입니다. 대화 상태는 아래 그림과 같이 미리 정의된 J(45)개의 Slot S마다 현재 턴까지 의도된 Value를 추론하여 (S, V)와 같은 페어의 집합(B)으로 표현될 수 있습니다. (* 이 때, 현재까지 의도되지 않은 정보(Slot)는 "none"이라는 특별한 Value를 가지게 되고, 아래 B에서 생략되어 있습니다.)

	![img](https://cphinf.pstatic.net/mooc/20210418_27/1618720176016yYvRl_PNG/mceclip1.png)

	우리가 학습 시킬 모델은 위 그림과 같이 매 턴마다 알맞은 Dialogue State B_t를 추론해야 합니다. 이런식으로 추론된 대화 상태는 외부 데이터베이스에 쿼리를 날려서 적절한 결과를 가지고 그 다음 시스템 답변을 만들어 내는데 사용됩니다. 즉, 대화 상태를 제대로 추적하지 못한다면 적절한 답변을 할 수가 없겠죠! 여러분의 모델이 과연 대화의 상태를 잘 추적할 수 있는지 알아보도록 합시다!

 

2. 데이터셋 개요

	Wizard-of-Seoul 데이터는 여러 대화의 리스트를 가지고 있습니다.

	하나의 대화는 아래와 같은 포맷을 가집니다.

	![img](https://cphinf.pstatic.net/mooc/20210418_200/16187198957432G0Q5_PNG/mceclip0.png)

	통계치는 아래와 같습니다.

	train_dials.json: 7000개의 대화 (label 포함)

	dev_dials.json: 1000개의 대화 (label 포함 / public test set)

	test_dials.json: 1000개의 대화 (label 미포함 / private test set)

	ontology.json: Ontology-based DST model을 위한 pre-defined ontology입니다.

	각 대화를 공통적으로 전처리하기 위해, `data_utils.py` 를 사용합니다. baseline 코드를 참조해주세요.

 

3. 종료 안내

	5/20 목요일 오후 7시에 대회는 종료됩니다.

 

대회 관련 더 자세한 사항은 AI stages에서 확인해 주시길 바랍니다.

감사합니다.

