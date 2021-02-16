# (3강) Recurrent Neural Network and Language Modeling

자연어 처리 분야에서 **Recurrent Neural Network(RNN)**를 활용하는 다양한 방법과 이를 이용한 **Language Model**을 학습합니다. RNN은 단어간 순서를 가진 문장을 표현하기 위해 자주 사용되어 왔습니다. 이러한 RNN 구조를 활용해 다양한 NLP 문제를 정의하고 학습하는 방법을 소개합니다. Language Model은 이전에 등장한 단어를 condition으로 다음에 등장할 단어를 예측하는 모델입니다. 이전에 등장한 단어는 이전에 학습했던 다양한 neural network 알고리즘을 이용해 표현될 수 있습니다. 이번 시간에는 RNN을 이용한 character-level의 language model에 대해서 알아봅니다.

RNN을 이용한 Language Model에서 생길 수 있는 초반 time step의 정보를 전달하기 어려운 점, gradient vanishing/exploding을 해결하기 위한 방법 등에 대해 다시 한번 복습할 수 있는 시간이 됐으면 합니다.

**Further Reading**

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [CS231n(2017)_Lecture10_RNN](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf)

-----

## Vanilla RNN and Type of RNN

RNN의 기본 구조와 RNN 계열 신경망의 종류에 관한 내용은 앞서 공부했던 내용과 중복되기에 생략하도록 하겠습니다. 

> 관련 내용에 대한 [포스팅](https://ydy8989.github.io/2021-02-04-rnn/)에서 확인하기.



## Character-level Language Model

언어모델 : 문자열이나 단어들의 순서를 바탕으로 다음 단어를 맞추는 task를 말한다. 

- Example of training sequence "hello"

	- Vocabulary : [h, e, l, o]

	- Example training sequence: “hello”

	- $ℎ_t = tanh(𝑊_th_{𝑡−1} + 𝑊_{xh}x_t + 𝑏)$ 를 통해서 Hidden layer 계산한다

	- output layer에서 $logit=W_{hy}h_t+b$로 계산된다. 

	- logit이라고 표현한 이유는, 다음 단어 혹은 character로 예측해야하는 

	- 각 time step에서 선형 변환을 통해 계산된 output layer는 softmax를 통과해 output을 도출하게 된다. 

	- 첫번째 timestep을 예로 들면, output layer의 output인 [1.0, 2.2, -3.0, 4.1]은 'e'에 해당하는 [0, 1, 0, 0]에 fitting되도록 학습한다. 

		![image](https://user-images.githubusercontent.com/38639633/108020955-e10f9c00-7060-11eb-864e-f63e5405dcc1.png)



### BPTT(Backpropagation through time)

![image](https://user-images.githubusercontent.com/38639633/108021388-cbe73d00-7061-11eb-85c0-4021e2c142c5.png)

- Forward through entire sequence to compute loss, then backward through entire sequence to compute gradient

- 위와 같이 loss를 계산할 때 한번에 계산한다면, 많은 리소스 제한이 걸리기 때문에 **truncated bptt**기법을 사용하게 된다. 

- Carry hidden states forward in time forever, but only backpropagate for some smaller number of steps.

  ![image](https://user-images.githubusercontent.com/38639633/108021514-1072d880-7062-11eb-9104-c44e94952479.png)



# Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)

이 부분에 대한 내용 역시 앞선 [포스팅](https://ydy8989.github.io/2021-02-04-rnn/)과 중복되므로 생략한다.

## LSTM

![image](https://user-images.githubusercontent.com/38639633/108021886-cc340800-7062-11eb-9ee8-9d2f41fe8374.png)

- Long short-term memory
  - i: Input gate, Whether to write to cell
  - f: Forget gate, Whether to erase cell
  - o: Output gate, How much to reveal cell
  - g: Gate gate, How much to write to cell
- 위 그림의 오른쪽 파란색 박스로 이해하자. 

## GRU

- Uninterrupted gradient flow!

	![image](https://user-images.githubusercontent.com/38639633/108022303-a6f3c980-7063-11eb-8edc-4026c482b8f4.png)

- `+`연산은 gradient를 복사해주는 역할을 한다. 

- 오리지날 RNN에 비해 Gradient를 조금 더 오래 보존할 수 있다. 



