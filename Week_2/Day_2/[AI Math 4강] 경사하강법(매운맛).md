# [AI Math 4강] 경사하강법(매운맛)

**경사하강법 기반의 선형회귀 알고리즘**에 대해 설명합니다.

경사하강법의 단점을 보완하는 **확률적 경사하강법(stochastic gradient descent)**을 소개합니다.

 

2강에서 배웠던 무어-펜로즈 역행렬을 활용한 선형회귀분석과 비교하여, 선형 모델 외에도 적용 가능한 경사하강법-선형회귀분석 방법을 설명합니다. 이 때 나오는 경사하강법 알고리즘 수식을 정확히 이해하고 넘어가면 좋겠습니다.

 

그리고 딥러닝에서 경사하강법이 가지는 한계를 설명하고, 이를 극복하기 위한 확률적 경사하강법을 소개합니다. 

확률적 경사하강법은 오늘날 딥러닝 학습에 널리 활용되는 방법이므로 충분히 공부하시고 넘어가시기 바랍니다.

 

# 선형회귀분석 복습

- `np.linalg.pinv`를 이용하면 데이터를 선형모델로 해석하는 선형회귀식을 찾을 수 있다. 
- 무어-펜로즈를 통한 근사
  - ![image-20210126200304410](https://user-images.githubusercontent.com/38639633/105856542-b49dcc80-602c-11eb-8939-c69467f11fea.png)
- 이번에는 경사하강법을 활용한 선형모델을 찾아보자. 



# 경사하강법으로 선형회귀 계수 구하기

- 선형회귀의 목적식은 $||y-X\beta||_2$이고 이를 최소화하는 $\beta$를 찾아야 하므로 다음과 같은 그래디언트 벡터를 구해야한다. 

  
  $$
  \nabla_\beta||y-X\beta||_2=(\part_{\beta_1}||y-X\beta||_2, \dots,\part_{\beta_d}||y-X\beta||_2)\\
  \begin{align}
  \part_{\beta_k}||y-X\beta||_2 &= \part_{\beta_k}\left\{\frac{1}{n}\sum^n_{i=1}\left(y_i-\sum^d_{j=1}X_{ij}\beta_j\right)\right\}\\
  &=-\frac{X_k^\top(y-X\beta)}{n||y-X\beta||_2}
  \end{align}
  $$
  

- 선형회귀의 목적식은 $||y-X\beta||_2$이고 이를 최소화하는 $\beta$를 찾는 것이므로, 위의 식
  $$
  \part_{\beta_k}||y-X\beta||_2 = -\frac{X_k^\top(y-X\beta)}{n||y-X\beta||_2}
  $$
  에서 모든 $k=1, \dots, d$에 해당하는 그래디언트 벡터를 구해야 한다.
  $$
  \begin{align}
  \nabla_\beta||y-X\beta||_2&=(\part_{\beta_1}||y-X\beta||_2, \dots,\part_{\beta_d}||y-X\beta||_2)\\
  &=\left(-\frac{X_{.1}^\top(y-X\beta)}{n||y-X\beta||_2},\dots,-\frac{X_{.d}^\top(y-X\beta)}{n||y-X\beta||_2}\right)\\
  &=-\frac{X^\top(y-X\beta)}{n||y-X\beta||_2}
  \end{align}
  $$

- 이제 목적식을 최소화하는 $\beta$를 구하는 경사하강법 알고리즘은 다음과 같다. 
  $$
  \beta^{t+1}\leftarrow \beta^{(t)} - \lambda\nabla_\beta||y-X\beta^{(t)}||
  $$

- 최소화하는 과정에서 $\beta$의 제곱을 최소화하는 값을 찾는 것과, $\beta^2$을 최소화하는 값을 찾는 것은 같다. 

- Pseudo

  ```
  input : X, y, lr, T, output : beta
  #norm : l2 norm을 계산하는 함수
  # lr : 학습률, T: 학습횟수
  for t in range(T):
  	error = y - X@beta
  	grad = -transpose(X)@error
  	beta = beta-lr*grad
  ```



# 경사하강법은 만능?

- 이론적으로 경사하강법은 미분가능하고 볼록(convex)한 함수에 대해서 적절한 학습률과 학습횟수를 선택했을 때 수렴이 보장되어 있다. 

- 특히, 선형회귀의 경우 목적식 $||y-X\beta||_2$는 회귀계수 $\beta$에 대해 볼록함수이므로 알고리즘을 충분히 돌리면 수렴이 보장된다. 

  ![image-20210126212537146](https://user-images.githubusercontent.com/38639633/105856550-b5cef980-602c-11eb-8cdc-b9b200a34352.png)

- 하지만 비선형회귀 문제의 경우, 목적식이 non convex할 수 있으므로 수렴이 보장되어 있지는 않다. 

  ![image-20210126212600726](https://user-images.githubusercontent.com/38639633/105856553-b7002680-602c-11eb-80d5-bec9d29a5502.png)

  

# 확률적 경사하강법

- 확률적 경사하강법은 모든 데이터를 사용해 업데이트하는 대신 <u>데이터 한개 또는 일부 활용하여 업데이트</u> 합니다. 
- 볼록이 아닌 목적식은 SGD를 통해 최적화할 수 있다. 
- SGD는 데이터의 일부를 가지고 파라미터를 업데이트하므로 연산자원을 좀 더 효율적으로 활용하는데 도움이 된다. 

## 확률적 경사하각법 : 미니배치의 연산

- 경사하강법은 전체데이터 $\mathscr{D}=(\mathbf{X},y)$를 가지고 목적식의 그래디언트 벡터임 $\nabla_\thetaL(\mathscr{D},\theta)$를 계산한다. 

- 반면에 SGD는 미니배치인 $\mathscr{D}_{(b)}=(\mathbf{X}_{(b)},y_{(b)})\subset \mathbf{D}$를 가지고 그래디언트 벡터를 계산한다. 

  ![image-20210126213633199](https://user-images.githubusercontent.com/38639633/105856559-ba93ad80-602c-11eb-92ee-02cf1f5767d3.png)

  > 단, 미니배치는 확률적으로 선택되므로 목적식의 모양이 바뀌며, 이로인해 그래디언트의 벡터가 달라진다. 

- Non-convex한 함수에서도 사용이 가능하므로, <u>머신러닝에 효율적이다.</u>

  ![image-20210126213633199](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210126213633199.png)

  

## 확률적 경사하강법의 원리 : 하드웨어

- 한번에 넣을 시 하드웨어적 한계점도 존재한다.
- 미니배치를 넣을 시 적은 데이터로 업데이트하며 자원 및 속도를 절약할 수 있다. 