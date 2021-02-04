# [DLBasic] Sequential Models - RNN

**강의 소개**

이번 강의는 이론, 실습, 과제로 구성됩니다.

주식, 언어와 같은 Sequential data와 이를 이용한 **Sequential model의 정의와 종류**에 대해 배웁니다.
그 후 딥러닝에서 sequential data를 다루는 **Recurrent Neural Networks 에 대한 정의와 종류**에 대해 배웁니다.

 

 

**Further Question**

- LSTM에서는 Modern CNN 내용에서 배웠던 중요한 개념이 적용되어 있습니다. 무엇일까요?
- Pytorch LSTM 클래스에서 3dim 데이터(batch_size, sequence length, num feature), `batch_first` 관련 argument는 중요한 역할을 합니다. `batch_first=True`인 경우는 어떻게 작동이 하게되는걸까요?



# Sequential Models

주식, 언어와 같은 Sequential data와 이를 이용한 **Sequential model의 정의와 종류**에 대해 배웁니다. 그 후 딥러닝에서 sequential data를 다루는 **Recurrent Neural Networks 에 대한 정의와 종류**에 대해 배웁니다.

**Further Question**

- LSTM에서는 Modern CNN 내용에서 배웠던 중요한 개념이 적용되어 있습니다. 무엇일까요?
- Pytorch LSTM 클래스에서 3dim 데이터(batch_size, sequence length, num feature), `batch_first` 관련 argument는 중요한 역할을 합니다. `batch_first=True`인 경우는 어떻게 작동이 하게되는걸까요?

<br>



## Sequential Model

**Markov model (first-order autoregressive model)**

- MDP(Markovian Decision Property) : 현재의 결과는 바로 직전 과거에만 영향을 받는다.
- 이는 현실의 많은 데이터데 적용되지 않는 모델이다. 

<br>



## Recurrent Neural Networks

RNN 모델은 앞서 언급했으므로 생략하기로 한다.

<br>



## Long Short Term Memory

![image-20210204134146206](https://user-images.githubusercontent.com/38639633/106863071-5b5c2a00-670b-11eb-98c0-e1b1f0bd9991.png)

LSTM의 구조는 위와 같다.  세부적인 gate에 대한 내용을 살펴보자

<br>



**Forget Gate**

![image-20210204134622316](https://user-images.githubusercontent.com/38639633/106863073-5bf4c080-670b-11eb-91a4-a0f07c42e89f.png)

Decide whish information to **throw** away  
어떤 정보를 잃어버릴지 결정한다.   

$$
f_t = \sigma(W_f\cdot\left[h_{t-1}, x_t\right]+b_f)
$$

- 현재의 입력 $x_t$와 이전의 output $h_{t-1}$를 input으로 받는다. 
- 결국 Sigmoid($\sigma$)를 통과하기 때문에 항상 0~1사이의 값을 받는다. 

<br>



**Input Gate**

![image-20210204135448884](https://user-images.githubusercontent.com/38639633/106863074-5bf4c080-670b-11eb-8bf4-5b18d1fc9be0.png)

Decide which information to **store** in the cell state  
정보중에 어떤 것을 올릴지 말지를 결정한다.   

$$
i_t = \sigma(W_f\cdot\left[h_{t-1}, x_t\right]+b_i)\\
\tilde{C} = tanh(W_C\cdot\left[h_{t-1}, x_t\right]+b_C)
$$

- 궁극적으로는 $\tilde{C}$가 현재 정보와 이전 출력값을 가지고 만드는 cell state의 예비군이다.

<br>



**Update Cell**

![image-20210204135803194](https://user-images.githubusercontent.com/38639633/106863076-5c8d5700-670b-11eb-8621-4f81847a2db0.png)

Update the cell state  

$$
i_t = \sigma(W_i\cdot\left[h_{t-1}, x_t\right]+b_i)\\
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
$$

- forget gate의 output과 input gate의 output을 취합해서 현재 정보 기준으로 새로운 cell state를 업데이트 한다. 

<br>



**Output Gate**

![image-20210204160322968](https://user-images.githubusercontent.com/38639633/106863077-5c8d5700-670b-11eb-87a2-8659c11c45a7.png)

Make output using the updated cell state  
앞서 설명한 **update cell**을 이용해 마지막으로 output을 출력한다.   

$$
o_t = \sigma(W_o\cdot\left[h_{t-1}, x_t\right]+b_o)\\
h_t = o_t * tanh(C_t)
$$


결론적으로 이 네 가지 gate들을 조합하여 LSTM을 구성하게 된다. 

<br>



## Gated Recurrent Unit

![image-20210204160810542](https://user-images.githubusercontent.com/38639633/106863078-5d25ed80-670b-11eb-8be1-f0f0fc785874.png){:.center}

$$
\begin{align}
&z_t = \sigma(W_z\cdot\left[h_{t-1},x_t\right])\\
&r_t = \sigma(W_r\cdot\left[h_{t-1},x_t\right])\\
&\tilde{h}_t = \text{tanh}(W\cdot\left[r_t*h_{t-1}, x_t\right]) \\
&h_t = (1-z_t)*h_{t-1} + z_t*\tilde{h}_t
\end{align}
$$


Simpler architecture with two gates(`reset gate` and `update gate`)  
No `cell state`, just `hidden state`.

- cell state가 없어짐으로써 output gate가 하나 줄었다. 대신, reset gate와 update gate가 있다. 
- 항상 그런 것은 아니지만, 몇몇 task에서 LSTM을 앞서는 모습을 종종 볼 수 있다. 

<br>