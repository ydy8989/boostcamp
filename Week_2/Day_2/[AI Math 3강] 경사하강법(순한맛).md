# [AI Math 3강] 경사하강법(순한맛)

미분의 개념과 **그래디언트 벡터**에 대해 설명합니다.

**경사하강법**의 알고리즘과 실제 코드에서의 구현을 보여줍니다.

 

접선의 기울기를 이용해서 함수의 최솟값으로 점을 이동시키는 원리를 알면 이를 바탕으로 경사하강법의 알고리즘과 수식을 이해할 수 있습니다.

특히 변수가 벡터인 경우, 편미분을 통해서 구한 그래디언트 벡터를 통해 d-차원으로 경사하강법을 확장할 수 있다는 개념을 확실하게 잡고 가시기 바랍니다.



# 미분

- Differentiation(미분)은 변수의 움직임에 따른 함수값의 변화를 측정하기 위한 방법
  $$
  f'(x)=\lim_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h}\\
  f(x) = x^2+2x=3\\
  f'(x)=2x + 2
  $$

- `sympy.diff` 를 통해 계산할 수 있다

  - ```python
    import sympy as sym
    from sympy.abc import x
    
    sym.diff(sym.poly(x**2 + 2*x + 3),x)
    ```

# 1. 미분의 기하학적 의미(그림)

- 미분은 함수$$f$$의 주어진 점 $$(x,f(x))$$에서의 **<u>접선의 기울기</u>**를 의미한다.

  ![image-20210126183651460](https://user-images.githubusercontent.com/38639633/105856533-b10a4580-602c-11eb-8ab7-3ad935e38ca8.png)

  > h를 0으로 수렴시키면 결국 점$$x$$에서의 기울기 값이 되는데, 이 값을 미분값이라고 한다.

- 한 점에서 접선의 기울기를 알면 어느 방향으로 점을 움직여야 함수값이 `증가`하는지 `감소`하는지 알 수 있다. 

- 기울기의 음/양을 통해 어느 방향으로 이동해야 함수값이 커지고 / 작아지는지를 파악할 수 있다.

  - 경사하강법 vs 경사상승법 

- 경사상승/경사하강 방법은 극값(미분값이 0인 지점)에 도달하면 움직임을 멈춘다.



## 1.1. 경사하강법 : 알고리즘

- Pseudo

  ```
  Input : gradient, init, lr, eps, Output:var
  ---
  #gradient : 미분을 계산하는 함수
  #init: 시작점, lr: 학습률, eps: 알고리즘 종료조건
  var = init
  grad = gradient(var)
  while(abs(grad) > eps):
  	var = var - lr*grad
  	grad = gradient(var)
  ```



# 2. 변수가 벡터일 때

- 미분은 <u>변수의 움직임에 따른 함수값의 변화를 측정하기 위한 도구</u>로 최적화에서 많이 사용된다. 

- 벡터가 입력인 다변수 함수의 경우 <u>편미분(partial differentiation)</u>을 사용한다.

  
  $$
  \partial_{x_i}f(x)=\lim_{h\rightarrow0}\frac{f(\mathbb{x}+h\mathbb{e_i})-f(\mathbb{x})}{h}\\
  f(x, y)=x^2 + 2xy+3+cos(x+2y)\\
  \part_xf(x,y)=2x+2y-sin(x+2y)
  $$
  

- 각 변수 별로 편미분을 계산한 <u>그래디언트 벡터</u>를 이용하여 경사하강/경사상승법에 사용할 수 있다.

   
  $$
  \nabla f=(\part_{x_1}f,\part_{x_2}f,\cdots,\part_{x_d}f)
  $$

  > 앞서 사용한 미분값인 $$f'(x)$$ 대신 벡터$$\nabla f$$를 사용하여 변수 $$\mathbb{x}=(x_1, \dots, x_d)$$를 동시에 업데이트 가능합니다.

- 다차원 공간에서의 그래디언트 벡터의 모습(3차원)

  ![image-20210126193241252](https://user-images.githubusercontent.com/38639633/105856536-b23b7280-602c-11eb-97fa-02757a14e4d7.png)

  > 극소점으로 향하는 각 차원(x, y, z)의 기울기 변화를 그린 모습

## 2.1. 경사하강법 : 알고리즘

- Pseudo

  ```
  Input : gradient, init, lr, eps, Output:var
  ---
  #gradient : 그래디언트 벡터를 계산하는 함수
  #init: 시작점, lr: 학습률, eps: 알고리즘 종료조건
  var = init
  grad = gradient(var)
  while(norm(grad) > eps):
  	var = var - lr*grad
  	grad = gradient(var)
  ```

  > 알고리즘은 2차원에서의 미분과 동일하나, 벡터에서는 절대값 대신 norm을 사용하여 종료조건을 설정한다.

  