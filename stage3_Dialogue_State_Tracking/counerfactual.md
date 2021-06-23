# counerfactual

>이유경
>(DST 강의 멘토 / Korea Univ.)



## Counterfactual

- counterfactual은 반 사시적 서술이라는 의미를 가진 단어
- 아주 예전부터 사용되는 단어지만, 머신러닝과 Casual learning에서 인과관계를 파악하기위해 자주 사용됨
	[https://christophm.github.io/interpretable-ml-book/counterfactual.html](https://christophm.github.io/interpretable-ml-book/counterfactual.html)
- NLP에서도 자주 활용되며, 최근 연구들은 Augmentation을 위해 자주 적용되었음
	LEARNING THE DIFFERENCE THAT MAKES A DIFFERENCE WITH COUNTERFACTUALLY-AUGMENTED DATA 
	- [https://arxiv.org/pdf/1909.12434.pdf](https://arxiv.org/pdf/1909.12434.pdf)
- EXPLAINING THE EFFICACY OF COUNTERFACTUALLY AUGMENTED DATA
	- [https://arxiv.org/pdf/2010.02114.pdf](https://arxiv.org/pdf/2010.02114.pdf)



## COCO: CONTROLLABLE COUNTERFACTUALS FOR EVALUATING DIALOGUE STATE TRACKERS

>  [https://arxiv.org/pdf/2010.12850.pdf](https://arxiv.org/pdf/2010.12850.pdf)



- COCO-DST에서 Counterfactual이라는 개념을 사용함
- DST에서 다루는 Counterfactual은 무슨 의미를 가지는 걸까요 ?
	- DST는 특정 시나리오가 이미 고정되어 있음 (Slot meta가 미리 주어지기 때문)
	- DST 모델이 좋은 성능을 보인다 != 현실 상황에서 잘 작동한다
	- COCO-DST에서는 학습된 Slot meta를 벗어날 경우 DST모델이 좋은 성능을 보이지 못한다는 것을 문제 삼음 !
	- 보다 현실적인 시나리오를 잘 반영하기 위해 기존 데이터를 Counterfactual 으로 변경



**왜? 이렇게할까?**

- 기존 데이터를 Counterfactual goal을 달성하기 위해 변경하게 될 경우 새로운 데이터를 Scratch부터 생성하지 않아도 비교적 간단한 방법으로 풍부한 데이터를 만들어 낼 수 있음
- Counterfactual dialogue를 통해 기존 데이터를 기준으로 반 사실적인 데이터를 생성하면서, 현실에 충분히 등장할 수 있는 데이터를 만들어 함께 학습해보자 !
- COCO-DST는 counterfactual dialogue를 만들어 모델의 성능을 향상시키지만 캠퍼분들께 Counterfactual dialogue를 생성하여 모델을 학습하라는 의미가 아닙니다 !
- Counterfactual dialogue를 만들면 내 모델은 성능도 좋고, 실제 상황에서도 잘 작동할 것

