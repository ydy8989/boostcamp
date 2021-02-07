---
layout: post
title: 부스트캠프 AI Tech - Generative Models
subtitle: Auto-regressive model, GAN으로 가기 위한 관문
thumbnail-img : https://user-images.githubusercontent.com/38639633/107111447-8ec9c080-6893-11eb-8961-be636107b65d.png
gh-repo: ydy8989/ydy8989.github.io
gh-badge: [follow]
categories: [BOOSTCAMP]
tags: [boostcamp]
comments: true
---

머신러닝, 딥러닝에서 의미하는 **Generative model이 무엇인지**, 이를 이해하기 위한 **기본적인 통계 이론**, **다양한 Generative model**의 아이디어, 구조에 대해 배웁니다.
수식적으로 어려움이 있을 수 있습니다.

<br>

# Introduction

Richard Feynman  

> "*What i cannot create, I do not understand*"



Generative model은 단순히 이미지를 만들고 텍스트를 생성하는 것만의 의미를 갖지 않는다. 

<br>

# Learning a Generative model

Suppose we are give images of dogs.

We want to learn a probability distribution $P(x)$ such that 

- **Generation** : If we sample $x_{new}\sim p(x), x_{new}$ should look like a dog(`sampling`)
- Density estimation : $p(x)$ should be high if $x$ looks like a dog, and low otherwise(`anomaly detection`)
	- Also known as, `explicit`(명백한) models
	- 입력이 주어졌을 때, 원하는 이미지가 '아닌' 것을 판단할 수 있는. 
	- 엄밀한 의미에서는 Generative model은 Discriminator를 포함하고 있다. 
- **Unsupervised representation learning** : We should be able to learn what these images have in common, e.g., ears, tail, etc(`feature learning`)

Then, how can we represent $p(x)$??

<br>

## Basic Discrete Distributions

1. Bernoulli distribution : coin flip
	1. D = {Heads, Tails}
	2. Specify $P(X = \text{Heads}) = p$. Then $P(X = \text{Tails}) = 1 − p$.
	3. Write : $X\sim$ Ber($p$)
2. Categorical distribution : m-sided die
	1. D = {1, $\dots$, m}
	2. Specify $P(X = i) = p_i$, such that $\sum^m_{i=1}P_i$=1.
	3. Write : $X\sim$ Cat($p_1,\cdots, p_m$)

<br>

### Example 1

Modeling an RGB joint distribution (of a single pixel)

- $\small(r, g, b) \sim p(R, G, B)$
- Number of cases?
	- 256 X 256 X 256
- How many parameters do we need to specify?
	- 255 X 255 X 255 

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/RGB_Cube_Show_lowgamma_cutout_b.png/1280px-RGB_Cube_Show_lowgamma_cutout_b.png){:width="60%"}{:.center}

<br>

### Example 2 - mnist

![image-20210205183012382](https://user-images.githubusercontent.com/38639633/107057698-145b5b00-6817-11eb-9e87-a6980bd15e2e.png){:.center}

Suppose we have $X_1, \dots, X_n$ of $n$ binary pixels (a binary image).

How many possible states?

$$
2\times2\times\cdots\times 2=2^n
$$


Sampling from $p(x_1, \dots, x_n)$ generates an image.

How many parameters to specify $p(x_1, \dots, x_n)$? 


$$
2^n -1
$$

어쨌든, 파라미터의 수가 많을 수록 점점 학습은 어려워지는 것이 통상적 이론!

<br>

## Structure Through Independence

파라미터를 줄이고자 시작한 가정 : 모든 픽셀이 서로 independent 하다면?
- What if  $\small X_1, \dots, X_n$ are independent, then
	
	$$
	p(x_1, \dots, x_n)= p(x_1)p(x_2)\cdots p(x_n)
	$$
- How many possible states?  : $2^n$
- How many parameters to specify $p(x_1, \dots, x_n)$? : $n$
- $2^n$ entires can be described by just $n$ numbers! But this **independence** assumption is too strong to model useful distributions.



결과적으로 픽셀을을 서로 독립하다고 가정한다면, $2^n$개의 파라미터가 $n$개로 줄어든다는 점은 엄청난 빅이득이다.

<br>

## Conditional Independence

따라서 이 두 파라미터의 중간 어딘가의 파라미터 수를 가진 distribution을 만들기 위한 몇 가지 트릭을 사용한다. 

- Three important rules
  - **Chain rule :**
  	
  	
  	$$
  	\small p(x_1, \dots, x_n)= p(x_1)p(x_2\vert x_1)p(x_3\vert x_1, x_2)\cdots p(x_n\vert x1, \cdots, x_{n-1})
  	\tag{1}
  	$$
  	
  - **Bayes' rule:**
  	
  	$$
  	\small P(x\vert y)=\frac{p(x,y)}{p(y)} = \frac{p(y\vert x)p(x)}{p(y)}
  	$$
  	
  - **Conditional independence:**
    - If $\small (x\perp y)\vert z$, then $p(x\vert y, z)=p(x\vert z)$ 
    - 만약 $z$가 주어졌을 때, $x$와 $y$가 independent 하다면, then $x$를 표현함에 있어 $y$는 상관이 없으므로 given $z$일 때, $y$와 상관없이 $x$의 확률로만 표현 가능하다.



위의 Chain Rule에서 각 파라미터의 수를 계산해보면  

- $\small p(x_1)$ : 1 parameter
- $\small p(x_2\vert x_1)$ : 2 parameter (one per $p(x_2\vert x_1 = 0)$ and one per $p(x_2\vert x_1 = 1)$)
- $\small p(x_3\vert x_1,x_2)$ : 4 parameter
- Hence, $1+2+2^2+\dots+2^{n-1}=2^n-1$, which is the same as before.

왜 파라미터 갯수가 같을까? 

- 결론적으로 아무것도 달라진게 없기 때문이다.
- now, suppose $\small \Rightarrow X_{i+1}\perp X_{1}, \dots, X_{i-1}\vert X_{i}$ , is called Markov assumption, then  

$$
p(x_1, \dots, x_n)= p(x_1)p(x_2\vert x_1)p(x_3\vert x_2)\cdots p(x_n\vert x_{n-1})
  \tag{2}
$$

- 위의 식 (1)이 식 (2)와 같은 이유는 독립적이라는 가정때문이다. 
- How many parameters?

$$
2n-1
$$

- Hence, by leveraging the Markov assumption, we get exponential reduction on the number of parameters.
- **Auto-regressive models** leverages this `conditional independency`.

<br>

# Auto-regressive Model

![image](https://user-images.githubusercontent.com/38639633/107035226-321ac700-67fb-11eb-9dc4-d3e08658702d.png)

- Suppose we have 28 by 28 binary pixels.
- Our goal is to learn $\small p(x) = p(x_1, \dots, x_784)$ over $$\small x\in \left\{0,1\right\}^{784}$$
- How can we parametrize $\small P(x)$?
	- Chain rule을 사용하여 결합확률분포(joint distribution)의 인자로 나타내면
	- $\small p(x_{1:784}) = p(x_1)p(x_2\vert x_1)p(x_3\vert x_{1:2})\cdots$
	- This is called an `autoregressive model` : 
		- 즉, autoregressor 모델은 바로 직전 뿐만아니라 이전의 모든 픽셀(데이터)이 dependent 한 것도 autoregressive model이라고 한다.
	- Note that we need `an ordering` of all random variables. 
		- 픽셀값에 순서를 매겨야할 필요성이 있다. 
		- 어떤식으로 conditional independency를 주느냐에 따라서 전체 스트럭쳐가 달라진다. 
		- 방금 우리의 예제는 markov assumption이었다. $\small \Rightarrow$ joint distribution을 어떻게 쪼개느냐가 중요

<br>

## NADE : Neural Autoregressive Density Estimator

![image](https://user-images.githubusercontent.com/38639633/107041537-fcc6a700-6803-11eb-8c19-6fd74f90aa0e.png){:.center}



- the probability distribution of **i**-th pixel is
	- $\small p(x_i\vert x_{1:i−1})=\sigma(\alpha_i h_i+b_i)$ where $$\small h_i=\sigma(W_{_<i}x_{1:i−1}+c)$$	
- i번째 픽셀을 첫번째부터 i-1번째 픽셀에 dependent 하게 한다
- neural network 입장에서는 입력차원이 계속 변한다
	- 그래서 weight가 계속 커진다.
	- 즉 3번째 픽셀에 대한 확률분포를 만들때는 1번째와 2번째 총 2개의 입력을 받는 weight가 필요한 반면
	- 100번째 필섹에 대한 확률분포를 만들때는 99개의 입력을 받는 weight가 필요하다.
- **NADE** is an **explicit** model that can compute the **density** of the given inputs
	- 단순히 generate만할 수 있는 것이 아니라 임의의 784개의 binary vector가 주어지면 이에 대한 확률을 계산할 수 있다
	- how about **implicit**?
		- 당연히 반대로 generate만 할 수 있는 모델을 의미한다. (확률 계산 불가능!)
- how can we compute the density of the given images?
	- suppose we have a binary image with 784 binary pixels, $$\small \left\{x_1,x_2,\dots, x_{784}\right\}$$
	- then, the joint probability is computed by
		- $\small p(x_1,...,x_{784})=p(x)p(x_2\vert x_1)\dots p(x_{784}\vert x_{1:783})$
		- where each conditional probability $\small p(x_i\vert x_{1:i−1})$ is computed independently
- In case of modeling continuous random variables, **a mixture of Gaussian** can be used  
	
	> ~~조만간에 [`논문`](https://arxiv.org/abs/1605.02226) 리뷰를 진행해야겠다.~~ 

<br>

## Pixel RNN

이미지의 픽셀을 만들고 싶은 모델임
- We cal alse use RNNs to define an auto-regressive model.
- For example, for an $n\times n$ RGB image,
	- $p(x)=\prod_{i=1}^{n^2} p(x_{i,R}\vert x_{<i}) p(x_{i,G}\vert x_{<i},x_{i,R})p(x_{i,B}\vert x_{<i},x_{i,R},x_{i,G})$
	- $p(x_{i,R}\vert x_{<i})$: Prob. i-th R
	- $p(x_{i,G}\vert x_{<i},x_{i,R})$ : Prob. i-th G
	- $p(x_{i,B}\vert x_{<i},x_{i,R},x_{i,G})$ : Prob. i-th B
- 차이점 
	- 앞에서는 fc-layer를 통해서 만들었다.
	- RNN을 통해서 generate 했다는 점이 차이점이 있다. 그 중에서 다시 두 가지 방식으로 **ordering**하는 방식이 나뉘게 된다. 
		- RowLSTM
		- Diagonal BiLSTM

![image](https://user-images.githubusercontent.com/38639633/107055560-c5142b00-6814-11eb-94a2-f87bb4d4cbb2.png){:.center}



> 이어서 다음 [**`포스팅`**](https://ydy8989.github.io/2021-02-05-gan2/)에서는 VAE를 비롯한 variational model과 대표적 생성모델 GAN에 대한 내용이 이어집니다.

