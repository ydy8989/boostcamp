# [AI Math 7강] 통계학 맛보기

모수의 개념과 모수를 추정하는 방법으로 **최대가능도 추정법**을 소개합니다.

정규분포, 카테고리분포에서의 예제로 최대가능도 추정법을 설명합니다.

 

표본분포와 표집분포, 가능도(likelihood)와 확률 등 헷갈릴 수 있는 개념들이 많이 소개되므로 각각의 정확한 의미와 차이점을 충분히 공부하고 넘어가시기 바랍니다.

 

최대가능도 추정법을 통해서 정답에 해당하는 확률분포와 모델이 추정하는 확률분포의 거리를 최소화함으로써 모델을 학습시킬 수 있으며, 이 원리는 딥러닝/머신러닝에서 아주 유용하게 사용되기 때문에 확실하게 이해하셨으면 좋겠습니다.

---

**Further Question**

1. 확률과 가능도의 차이는 무엇일까요? (개념적인 차이, 수식에서의 차이, 확률밀도함수에서의 차이
2. 확률 대신 가능도를 사용하였을 때의 이점은 어떤 것이 있을까요?

3. 다음의 code snippet은 어떤 확률분포를 나타내는 것일까요? 해당 확률분포에서 변수 theta가 의미할 수 있는 것은 무엇이 있을까요?

```python
import numpy as np
import matplotlib.pyplot as plt
theta = np.arange(0, 1, 0.001)
p = theta ** 3 * (1 - theta) ** 7
plt.plot(theta, p)
plt.show()
```

---

# 1. 모수가 뭐에요?

- 통게적 모델링은 적절한 가정 위에서 확률분포를 추정하는 것이 목표이며, 기계학습과 통계학이 공통적으로 추구하는 목표이다. 
- 유한한 개수의 데이터로부터 모집단의 분포를 알아내는 것은 불가능하므로, 근사적인 확률분포를 추정할 수 밖에 없다. 
  - 분포 자체를 맞춘다기보다 데이터와 추정 방법의 **<u>불확실성을 최소화하는 방향</u>**
- 데이터가 특정​ 확률분포를 따른다고 선험적으로(a priori) 가정한 후 그 분포를 결정하는 모수(**parameter**)를 추정하는 방법을 모수적(**parametric**) 방법론이라 한다.
- 특정 확률분포를 가정하지 않고, 데이터에 따라 모델의 구조 및 모수의 개수가 유연하게 바뀌면 **비모수 방법론**이라고 한다.

<br>



## 1.1. 확률분포 가정하기 : 예제

- 확률분포를 가정하는 방법 :
  - 히스토그램을 통해 모양 관찰
  - ex) 데이터가 n개의 이산적인 값을 가지는 경우 $\Rightarrow$ 카테고리 분포
- 기계적으로 확률분포 가정을 단정지으면 안되고, **데이터 생성 원리를 먼저 고려하는 것**이 원칙이다.

<br>

# 2. 데이터로 모수 추정

- 데이터의 확률분포를 가정했다면 모수를 추정할 수 있다. 

- 정규분포의 모수는 평균 $\mu$와 분산 $\sigma^2$으로, 이를 추정하는 통계량은 다음과 같다.:

  
  $$
  \bar{X} = \frac{1}{N}\sum^N_{i=1}X_i\\
  \mathbb{E}[\bar{X}]=\mu
  $$
  
  $$
  S^2 =\frac{1}{N-1}\sum^N_{i=1}(X_i-\bar{X})^2\\
  \mathbb{E}[S^2] = \sigma^2
  $$
  
- 표본 분산의 계산 시 $N-1$로 나노는 이유는 unbiased(불편)추정량을 구하기 위해서이다.

- 통계량의 확률분포를 표집분포라 부르며, 특히, 표본평균의 표집분포는 $N$이 커질수록 정규분포 $\mathcal{N}(\mu, \sigma^2/N)$를 따른다.

<br>

# 3. 최대가능도 추정

- 표본평균이나 표본분산은 중요한 통계량이지만 확률분포마다 사용하는 모수가 다르기 때문에 적절한 통계량이 달라지게 됩니다. 

- 이론적으로 가장 가능성이 높은 모수를 추정하는 방법 중 하나는 **최대가능도 추정법(Maximum Likelihood Estimation, MLE)**이다.
  $$
  \hat{\theta}_{MLE} = \operatorname*{argmin}_\theta L(\theta;\mathbf{x})=\operatorname*{argmin}_\theta P(\mathbf{x}\vert\theta)
  $$

  > likelihood func.는 모수 $\theta$를 따르는 분포가 $\mathbf{x}$를 관찰할 가능성을 뜻하지만, 확률로 해석하면 안된다. 
  >
  > 즉, 고정된 데이터(관찰된 데이터) $\mathbf{x}$가 주어진 상황에서, $\theta$를 변형시킴에 따라 값이 바뀌는 함수, 바꿔말하면 모수 $\theta$를 따르는 분포가 데이터 $\mathbf{x}$를 관찰할 가능성을 뜻하게 된다. 
  >
  > 확률밀도함수에서는 모수 $\theta$가 이미 알고 있는 상수계수고 $\mathbf{x}$가 변수다. 하지만 모수 추정 문제에서는 $\mathbf{x}$ 즉, 이미 실현된 표본값은 알고 있지만 모수 $\theta$를 모르고 있다. 이때는 반대로 $\mathbf{x}$를 이미 알고있는 상수계수로 놓고 $\theta$를 변수로 생각한다. 물론 함수의 값 자체는 변함없이 주어진 $\mathbf{x}$가 나올 수 있는 확률밀도다. 이렇게 **확률밀도함수에서 모수를 변수로 보는 경우에 이 함수를 가능도함수(likelihood function)**라고 한다. 같은 함수를 확률밀도함수로 보면 $P(\mathbf{x};\theta)$로 표기하지만 가능도함수로 보면 $L(\theta;\mathbf{x})$ 기호로 표기한다.

## 3.1. 로그가능도

- 데이터 집합 $\mathbf{X}$가 `독립적`으로 추출되었을 경우 로그가능도를 다음과 같이 나타낼 수 있다.
  $$
  L(\theta;\mathbf{X})=\prod^n_{i=1}P(\mathbf{x}_i\vert\theta)~~\Rightarrow~~\text{log}L(\theta;\mathbf{X})=\sum^n_{i=1}\text{log}P(\mathbf{x}_i\vert\theta)
  $$

  > 로그를 씌우는 과정을 통해 곱셉 연산을 덧셈으로 변환하여 최적화할 수 있다. 

<br>

## 3.2. 왜 로그 가능도?

- 로그가능도를 최적화하는 모수 $\theta$는 가능도를 최적화하는 MLE가 된다.\
- 데이터의 양이 수억 단위가 된다면 컴퓨터의 정확도로는 가능도를 계산하는 것이 불가능하다
- 데이터가 독립일 경우 곱셈을 덧셈으로 바꿀 수 있기 때문에 컴퓨팅 연산이 쉬워진다.
- 경사하강법을 사용할 시 연산량이 가능도의 경우 $\mathcal{O}(n^2)$에서 로그가능도의 $\mathcal{O}(n)$로 줄어든다.
- 대부분의 Loss func.의 경우 경사하강법을 사용하므로 **음의 로그가능도**를 최적화하게 된다. 

<br>

## 3.3. 추정법 예제 : 정규분포

정규분포의 확률밀도함수는 다음과 같다. 여기서 $x$는 스칼라 값이다.
$$
\begin{align}
p(x ; \theta ) = \mathcal{N}(x ; \mu, \sigma^2) = \dfrac{1}{\sqrt{2\pi\sigma^2}} \exp \left(-\dfrac{(x-\mu)^2}{2\sigma^2}\right)\\
\end{align}
$$
그런데 표본 데이터가 $x_1, \cdots, x_N$이 있는 경우, 모두 독립이므로 전체 확률밀도함수는 각각의 확률밀도함수의 곱과 같다. 
$$
\begin{align}
L(\mu;x_1, \cdots, x_N) = p(x_1, \cdots, x_N;\mu) = \prod_{i=1}^N  \dfrac{1}{\sqrt{2\pi\sigma^2}} \exp \left(-\dfrac{(x_i-\mu)^2}{2\sigma^2}\right)
\end{align}
$$
계산을 쉽게하기 위한 로그변환 후 로그가능도를 구하면 다음과 같다. 
$$
\begin{split} 
\begin{align}
\begin{aligned}
\log L 
&= \log p(x_1, \cdots, x_N;\mu)  \\
&= \sum_{i=1}^N \left\{ -\dfrac{1}{2}\log(2\pi\sigma^2) - \dfrac{(x_i-\mu)^2}{2\sigma^2} \right\} \\
&= -\dfrac{N}{2} \log(2\pi\sigma^2) - \dfrac{1}{2\sigma^2}\sum_{i=1}^N (x_i-\mu)^2
\end{aligned}
\end{align}
\end{split}
$$
이 확률밀도함수가 최대가 되는 모숫값을 찾기 위해서는 각각의 모수로 미분한 값이 0이 되어야 한다.
$$
\begin{split}
\begin{align}
\begin{aligned}
\dfrac{\partial \log L}{\partial \mu} 
&= \dfrac{\partial}{\partial \mu} \left\{ \dfrac{N}{2} \log(2\pi\sigma^2) + \dfrac{1}{2\sigma^2}\sum_{i=1}^N (x_i-\mu)^2  \right\} = 0 \\
\dfrac{\partial \log L}{\partial \sigma^2} 
&= \dfrac{\partial}{\partial \sigma^2} \left\{ \dfrac{N}{2} \log(2\pi\sigma^2) + \dfrac{1}{2\sigma^2}\sum_{i=1}^N (x_i-\mu)^2 \right\} = 0\\
\end{aligned}
\end{align}
\end{split}
$$
이 두 식을 풀면 주어진 데이터 표본에 대해 모수의 가능도를 가장 크게 하는 모수의 값을 구할 수 있다. 먼저 $\mu$에 대한 미분을 정리하면 다음과 같다.
$$
\begin{align}
\dfrac{\partial \log L}{\partial \mu}  = \dfrac{2}{2\sigma^2}\sum_{i=1}^N (x_i-\mu) = 0
\end{align}
$$

$$
N \mu = \sum_{i=1}^N x_i
$$

$$
\begin{align}
\mu = \dfrac{1}{N}\sum_{i=1}^N x_i = \bar{x}
\end{align}
$$

다음으로 $\sigma^2$에 대한 미분을 정리하면 다음과 같다.
$$
\begin{align}
\dfrac{\partial \log L}{\partial \sigma^2}  = \dfrac{N}{2\sigma^2} -  \dfrac{1}{2(\sigma^2)^2}\sum_{i=1}^N (x_i-\mu)^2  = 0
\end{align}
$$

$$
\begin{align}
\sigma^2  =  \dfrac{1}{N}\sum_{i=1}^N (x_i-\mu)^2 = \dfrac{1}{N}\sum_{i=1}^N (x_i-\bar{x})^2 = s^2
\end{align}
$$

따라서 결론은 다음과 같다. 

> **최대가능도 추정법에 의한 정규분포의 기대값은 표본평균과 같고, 분산은 (편향)표본분산과 같다.**



<br>



## 3.4. 추정법 예제 : 카테고리분포

카테고리 분포를 따르는 확률변수 $\mathbf{X}$로부터 독립적인 표본 $\left\{x_1, \dots, x_N\right\}$을 얻었을 때 최대가능도를 구해보자. 먼저 확률밀도함수는 각각의 확률질량함수의 곱과 같다. 
$$
\begin{align}
L(\mu_1, \cdots, \mu_K ; x_1,\cdots, x_N) = \prod_{i=1}^N \prod_{k=1}^K \mu_k^{x_{i,k}}
\end{align}
$$
이 식에서 $x_{i,k}$는 $i$번째 시행 결과인 $x_i$의 $k$번째 원소를 뜻한다. 

미분을 쉽게 하기 위해 로그 변환을 한 로그가능도를 구하면 다음과 같다. 
$$
\begin{split} 
\begin{align}
\begin{aligned}
\log L 
&= \log p(x_1, \cdots, x_N;\mu_1, \cdots, \mu_K)  \\
&= \sum_{i=1}^N \sum_{k=1}^K  \left( {x_{i,k}} \log\mu_k  \right) \\
&= \sum_{k=1}^K  \sum_{i=1}^N  \left(  \log\mu_k \cdot {x_{i,k}}\right) \\
&= \sum_{k=1}^K \left( \log\mu_k \left( \sum_{i=1}^N {x_{i,k}}   \right)  \right)
\end{aligned}
\end{align}
\end{split}
$$
$k$번째 원소가 나온 횟수를 $N_k$라고 표기하면
$$
N_k=\sum^N_{i=1}x_{i,k} ~\text{이고,}\\
\log L = \sum_{k=1}^K \left( \log\mu_k  \cdot N_k  \right)
$$
이다. 이때, 함수를 최대화하는 모수의 값을 찾아야 하는데 모수는 제한조건 
$$
\sum_{k=1}^K \mu_k = 1
$$
을 만족해야만 한다. 따라서 [`라그랑주 승수법`](https://untitledtblog.tistory.com/96)을 사용하여 로그가능도에 제한조건을 추가한 새로운 목적함수를 생각할 수 있다. 
$$
\begin{align}
J = \sum_{k=1}^K \log\mu_k N_k  + \lambda \left(1- \sum_{k=1}^K \mu_k \right)

\end{align}
$$
이 때 모수로 미분한 값이 0이 되는 값을 구하면 된다. 
$$
\begin{split}
\begin{align}
\begin{aligned}
\dfrac{\partial J}{\partial \mu_k} 
&= \dfrac{\partial}{\partial \mu_k} \left\{ \sum_{k=1}^K \log\mu_k N_k  + \lambda \left(1- \sum_{k=1}^K \mu_k\right)  \right\} = 0  \;\; (k=1, \cdots, K) \\
\dfrac{\partial J}{\partial \lambda} 
&= \dfrac{\partial}{\partial \lambda} \left\{ \sum_{k=1}^K \log\mu_k N_k  + \lambda \left(1- \sum_{k=1}^K \mu_k \right)  \right\} = 0 & \\
\end{aligned}

\end{align}
\end{split}
$$
이를 풀면 다음과 같이 모수를 추정할 수 있다.
$$
\begin{align}
\dfrac{N_1}{\mu_1}  = \dfrac{N_2}{\mu_2} = \cdots = \dfrac{N_K}{\mu_K} = \lambda

\end{align}
$$

$$
N_k = \lambda \mu_k
$$

$$
\sum_{k=1}^K N_k = \lambda \sum_{k=1}^K \mu_k  = \lambda = N
$$

$$
\mu_k = \dfrac{N_k}{N}
$$

따라서 결론은 다음과 같다.

> **최대가능도 추정법에 의한 카테고리분포의 모수는 각 범주값이 나온 횟수와 전체 시행횟수의 비율이다.**



<br>



# 4. 딥러닝에서의 최대가능도 추정

- 최대가능도 추정법을 이용해서 머신러닝 모델을 학습할 수 있다. 

- 딥러닝 모델의 가중치를 $\theta=(\mathbf{W}^{(1)}, \dots, \mathbf{W}^{(L)})$이라 표기했을 때 분류 문제에서 softmax 벡터는 카테고리분포의 모수를 모델링한다. 

- 원핫벡터로 표현한 정답레이블 $\mathbf{y} = (y_1, \dots, y_k)$를 관찰데이터로 이용해 확률분포인 softmax 벡터의 로그가능도를 최적화할 수 있다. 
  $$
  \hat{\theta}_{MLE} = \operatorname*{argmin}_\theta\frac{1}{n}\sum_{i=1}^n\sum_{k=1}^K y_{i,k}\text{log}(\text{MLP}_\theta(\mathbf{x}_i)_k)
  \tag{1}
  $$

  > 위 수식을 기억해두자

  

  <br>

  

# 5. 확률분포의 거리 구하기

- ML에서의 Cost function은 training시에 학습하는 확률분포와 데이터로부터 관찰되는 확률분포의 거리를 통해 유도한다. 

- 데이터 공간에 두개의 확률분포$P(\mathbf{x}),Q(\mathbf{x})$가 있다고 가정할 때 **두 확률분포 사이의 거리(distance)**는 다음의 세 가지 방식을 활용한다. 

  1. 총변동 거리(Total Variation Distance, TV)
  2. 쿨백-라이블러 발산(Kullback-Leibler Divergence, KL-Divergence)
  3. 바슈타인 거리(WassersteinDistance)

  

# 6. 쿨백 라이블러 발산

쿨백-라이블러 발산은 다음과 같이 정의한다. 

이산확률변수의 경우
$$
\mathbb{KL}(P\|Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$
연속확률변수의 경우
$$
\mathbb{KL}(P\|Q) = \int_{-\infty}^\infty p(x) \log \frac{p(x)}{q(x)} dx
$$
로 정의하며, 이때 $p,q$는 각 확률분포의 **확률밀도함수**를 의미한다.

한편, 쿨백라이블러는 다음과 같이 분해할 수 있는데,
$$
\mathbb{KL}(P\vert\vert Q) = −\mathbb{E}_{x\sim P(x)}[\text{log}Q(x)] + \mathbb{E}_{x\sim P(x)}[\text{log} P(x)]
$$
우변의 마이너스텀을 크로스엔트로피, 플러스텀을 엔트로피라고 부른다. 

중요한 점은 앞서 식 `(1)`의 $\hat{\theta}_{MLE}$는 분류문제에서 정답레이블을 $P$, 모델 예측을 $Q$라 두면 최대가능도 추정법은 쿨백-라이블러 발산을 최소화하는 것과 같다고 할 수 있다. 



### References

> https://datascienceschool.net/02%20mathematics/09.02%20%EC%B5%9C%EB%8C%80%EA%B0%80%EB%8A%A5%EB%8F%84%20%EC%B6%94%EC%A0%95%EB%B2%95.html#id14

