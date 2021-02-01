# [DLBasic] 뉴럴 네트워크 - MLP

신경망(Neural Networks)의 정의, Deep Neural Networks에 대해 배웁니다.

**신경망(Neural Networks)**

간단한 Linear neural networks 를 예시로 Data, Model, Loss, Optimization algorithm 을 정의해보는 시간을 가집니다.

**Deep Neural Networks**

Deep Neural Netowkrs란 무엇이며 Multi-layer perceptron와 같이 더 깊은 네트워크는 어떻게 구성하는지에 대해 배웁니다. 끝으로 **Pytorch를 이용하여 MLP 실습**을 합니다.

<br>



# 1. Neural networks

- "*위키피디아*" - 신경망은 동물의 뇌 구조로부터 생물학적으로 '애매하게' 영향을 받은 컴퓨팅 시스템이다.
- 신경망은 어떠한 함수를 모방하는 function approximator라고 정의할 수 있다.

<br>



# 2. Linear Neural Networks

- 선형 모델은 가장 간단한 예라고 할 수 있다. 

	![image-20210201224113479](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210201224113479.png)

- Data : $\mathcal{D}=\left\{(x_i, y_i)\right\}^N_{i=1}$

- Model : $\hat{y}=wx+b$ 

- Loss : loss = $\frac{1}{N}\sum^N_{i=1}(y_i-\hat{y}_i)^2$

- 어쨌든 목적은 $N$개의 데이터를 잘 표현할 수 있는 모델을 찾는 과정이다. 

- 나의 파라미터를 어떠한 `방향`으로 움직였을 때 더 작아지는 loss값을 찾고, 그 방향으로 바꾸는 과정이다. 
	$$
	\begin{align}
	\frac{\part{loss}}{\part{w}}
	&=\frac{\part}{\part{w}}\frac{1}{N}\sum^N_{i=1}(y_i-\hat{y}_i)^2\\
	&=\frac{\part}{\part{w}}\frac{1}{N}\sum^N_{i=1}(y_i-wx_i-b)^2\\
	&=-\frac{1}{N}\sum^N_{i=1}-2(y_i-wx_i-b)x_i
	\end{align}
	$$

- bias `b`도 마찬가지!
	$$
	\begin{align}
	\frac{\part{loss}}{\part{b}}
	&=\frac{\part}{\part{b}}\frac{1}{N}\sum^N_{i=1}(y_i-\hat{y}_i)^2\\
	&=\frac{\part}{\part{b}}\frac{1}{N}\sum^N_{i=1}(y_i-wx_i-b)^2\\
	&=-\frac{1}{N}\sum^N_{i=1}-2(y_i-wx_i-b)
	\end{align}
	$$

	> Gradien Descent

## 2.1. Multi dimensional input

- 다차원 인풋이 주어졌을 때에도 마찬가지이다. 

	![image-20210201225316050](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210201225316050.png)

- matrix로 나타낼 수 있다는 것은 서로 다른 두 차원을 연결(mapping) 혹은 선형변환 해줄 수 있음을 의미한다.

	![image-20210201230325194](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210201230325194.png)

	> $W^\top$과 $\mathbf{b}$를 통해 벡터공간 $\mathbf{x}$를 벡터공간 $y$로 선형변환 하는 모습

	

<br>



# 3. Beyond Linear Neural Networks

- 위와 같은 과정이 쌓이면(Stacked) 선형 변환을 여러번 겹친 모습이 된다. 

	![image-20210201230745408](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210201230745408.png)

- 하지만, 항상 선형변환이 가능한 것은 아니다. 

	> Nonlinear transform인 $\rho$
	>
	> $y = W^\top_2\mathbf{h}=W^\top_2\rho(W^\top_1)\mathbf{x}$

- 이러한 비선형 변환을 만족하기 위한 Activation functions들

	![image-20210201231207060](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210201231207060.png)

	



<br>



# 4. MLP(Multi-layer perceptron)

- 위와 같은 비선형 변환을 통해 레이어를 깊게 쌓는다. 
- 나머지는 코드로 알아보자.