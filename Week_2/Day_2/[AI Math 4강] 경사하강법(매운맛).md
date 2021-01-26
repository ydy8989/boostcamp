# [AI Math 4강] 경사하강법(매운맛)

**경사하강법 기반의 선형회귀 알고리즘**에 대해 설명합니다.

경사하강법의 단점을 보완하는 **확률적 경사하강법(stochastic gradient descent)**을 소개합니다.

 

2강에서 배웠던 무어-펜로즈 역행렬을 활용한 선형회귀분석과 비교하여, 선형 모델 외에도 적용 가능한 경사하강법-선형회귀분석 방법을 설명합니다. 이 때 나오는 경사하강법 알고리즘 수식을 정확히 이해하고 넘어가면 좋겠습니다.

 

그리고 딥러닝에서 경사하강법이 가지는 한계를 설명하고, 이를 극복하기 위한 확률적 경사하강법을 소개합니다. 

확률적 경사하강법은 오늘날 딥러닝 학습에 널리 활용되는 방법이므로 충분히 공부하시고 넘어가시기 바랍니다.

 

# 선형회귀분석 복습

- `np.linalg.pinv`를 이용하면 데이터를 선형모델로 해석하는 선형회귀식을 찾을 수 있다. 
- 무어-펜로즈를 통한 근사
  - ![image-20210126200304410](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210126200304410.png)
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
  &=\left(-\frac{X_{.1}^\top(y-X\beta)}{n||y-X\beta||_2},\dots,-\frac{X_{.d}^\top(y-X\beta)}{n||y-X\beta||_2}\right)
  \end{align}
  $$
  

