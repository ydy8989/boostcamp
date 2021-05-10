# (5강) Open-Vocab based DST Models

**강의 소개**

이번 강의에서는 DST 문제풀이의 또 다른 축으로 Ontology에 의존하지 않고 DST 문제를 푸는 Open-Vocab approach를 공부합니다.

Open-Vocab approach를 배우면서 기존의 Ontology-based approach와는 어떤 차이가 있는지를 배우고, 대표적인 Open-Vocab model인 TRADE에 대해 자세하게 알아봅니다.

TRADE가 competition의 베이스라인 코드인 만큼 이 모델에서 어떻게 DST 문제를 풀어갔고, 여러분은 이를 바탕으로 어떻게 문제를 풀어가면 좋을지 생각하며 공부해 보시길 바랍니다!

 

## Pitfalls of Ontology-based DST

predefined ontology-based dst model이 가지는 한계에 대해 알아보자

### Ontology-based model recap

DST as classification on th predefined ontology

![image](https://user-images.githubusercontent.com/38639633/116956580-8f29fa00-acd0-11eb-8761-e25707abe130.png)

- 미리 준비된 온톨로지들로부터 분류한다. 



### Pitfall of Ontology-based Model

Ontology-based DST의 특징

- ontology = Prior knowledge
- Ontology volume에 따른 complexity
- unseen slot value tracking의 어려움



**Prior Knowledge**

미리 준비된 ontology 내에서 분류를 진행하기 때문에 쉽다. 하지만, single domain이 아닌 multi domain에서는 어렵고 Volumen Complexity의 측면에서도 사이즈가 커질수록 정교하게 계산하기가 힘들다. 

![image](https://user-images.githubusercontent.com/38639633/116956764-0b244200-acd1-11eb-8377-96dd1e6f8705.png)

- Slot value의 후보가 많을 수록 정교한 비교가 힘들다. 



## Open-Vocabulary DST Model

Open-Vocab 기반 DST 모델에 대해 알아봅니다.

### Generation-based DST Recap

Decoder를 사용함으로써 conditional한 language modeling을 통해서 value가 등장할 확률을 vocab space에 표현하는 방식이다. 

![image](https://user-images.githubusercontent.com/38639633/116182240-dd7e4c80-a756-11eb-8093-267372591b1e.png)



### Open Vocabulary?

- Ontology의 존재를 가정하지 않는다.
	- Ontology Volume이 증가해도 복잡도가 증가 하지 않는다
	- Unseen value를 Tracking하기 용이하다
- Value에 대한 확률적인 표현을 Generation 혹은 Extraction(Like MRC)을 통해 한다.

![image](https://user-images.githubusercontent.com/38639633/116958165-a4088c80-acd4-11eb-9a5a-9aca20693ad0.png)

### Sequicity

[[paper] Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures (2018, Lei et al.)](Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures (2018, Lei et al.))

- 이전까지는 classification method로 다뤘지만, seq2seq 방식의 single framwork 하나로 구현하였다. 

	- $B$ : dialogue state

	- $R$ : 시스템 발화

	- $U$ : 유저 발화

	- 현재턴의 dialogue state는 이전 턴에 트래킹 되었던 $B_{t-1}, R_{t-1}, U_{t-1}$이 input으로 들어가게 된다. 

		![image](https://user-images.githubusercontent.com/38639633/117569108-f1369500-b0fe-11eb-8342-76b2b7681631.png)

	- eorleorl













**Further Reading**

- [Sequicity](https://www.aclweb.org/anthology/P18-1133/)
- [DST-Reader](https://www.aclweb.org/anthology/W19-5932/)
- [TRADE](https://www.aclweb.org/anthology/P19-1078/)

