# [DLBasic] Optimization

**강의 소개**

이번 강의는 이론, 실습, 과제로 구성됩니다.

**최적화와 관련된 주요한 용어**와 **다양한 Gradient Descent 기법**들을 배웁니다.


각 용어들의 의미에 대해 배우고, 기존 SGD(Stochastic gradient descent)를 넘어서 최적화(학습)가 더 잘될 수 있도록 하는 다양한 Gradient Descent 기법들에 대해 배웁니다.

**Further Questions**

- 올바르게(?) cross-validation을 하기 위해서는 어떻 방법들이 존재할까요? 
- Time series의 경우 일반적인 k-fold cv를 사용해도 될까요?
	- [TimeseriesCV](https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9)

**Further reading**

- [RAdam github](https://github.com/LiyuanLucasLiu/RAdam)
- [AdamP github](https://github.com/clovaai/AdamP)

---

# 1. Introduction

생떽쥐베리 *"언어는 잘못된 오해의 원천이다."*

용어를 통일하지 않는다면 새로운 지식을 습득하고 이해하기 어려울 수 있다. 

**Gradient Descent**

미분 가능한 함수를 1계 미분을 반복적으로 시도한다면 로컬 미니멀로 도달할 수 있다. 이 국소 최저값에 도달하기 위한 Optimization을 알아보기에 앞서 용어들을 알아보자



# 2. Important Concepts in Optimization

> Generalization
>
> Under-fitting vs over-fitting
>
> Cross validation
>
> Bias-variance tradeoff
>
> Bootstrapping
>
> Bagging and boosting



## 2.1. Generalization

**How well the learned model will behave unseen data**

Training error가 낮을 수록 성능이 좋은 것은 아니다. Test error가 어느 순간부터 올라가기 때문이다. 이 두 차이를 Generalization gap이라고 말한다. 

그렇다면 Generalization gap이 작을 수록 좋은 성능? No!

- Training error가 애초에 낮지 않으면 generalization gap이 작은 것이 무의미하다. 

![image-20210202113625150](https://user-images.githubusercontent.com/38639633/106569218-9b8aa380-6577-11eb-9a84-7a3c52335d5e.png)



## 2.2. Underfitting vs Overfitting

정말 원하는 모델은 overfitting과 underfitting 사이 그 어딘가에 있다. (이론적으로)

![image-20210202113856307](https://user-images.githubusercontent.com/38639633/106569222-9cbbd080-6577-11eb-8a14-f44109407685.png)



## 2.3. Cross-validation

**Cross-validation is a model validation technique for assessing how the model will generalize to an independent (test) data set.**

그렇다면 일반적인 모델을 만들기 위해서는 validation과 training 데이터를 각각 몇대몇으로 나눠야할까? 이를 해결하기 위한 방법론이 `cross validation`

![image-20210202114058783](https://user-images.githubusercontent.com/38639633/106569226-9d546700-6577-11eb-950c-1d04e22b98c6.png)

일반적으로 cross-validation을 시도하여 최적의 학습 파라미터를 찾고, 이 학습 파라미터를 고정한 상태에서 전체 데이터를 사용하여 학습한다. 



## 2.4. Bias and Variance

Bias : 일반적으로 데이터의 군집이 중심(원하는 목표)로부터 떨어진 정도를 의미한다.

Variance : 군집이 흩어진 정도를 의미한다. 

![image-20210202114733255](https://user-images.githubusercontent.com/38639633/106569229-9d546700-6577-11eb-93b6-d0afd36796d4.png)

### 2.4.1. Bias and Variance Trandeoff

> Given $\mathcal{D}=\left\{(x_i, t_i)\right\}^N_{i=1}$, where $t=f(x)+\epsilon$ and $\epsilon\sim \mathcal{N}(0,\sigma^2)$
>
> We can derive that what we are minimizing(cost) can be decomposed into three different parts: **bias**$^2$, **variance**, and **noise**.

$$
\begin{align}
\mathbb{E}[(t-\hat{f})^2]
&=\mathbb{E}[(t-f+f-\hat{f})^2]\\
&=\cdots\\
&=\mathbb{E}[(f-\mathbb{E}[\hat{f}]^2)^2]+\mathbb{E}[(\mathbb{E}[\hat{f}]-\hat{f})^2]+\mathbb{E}[\epsilon]
\end{align}
$$

노이즈가 껴있을 때에는 bias와 variance를 모두 줄이는 것은 힘들다. 



## 2.5. Bootstrapping

any test or metric that uses random sampling with replacement.

여러 샘플링 조합을 통해서 모델을 강건하게 만드는 방법을 의미한다. 

![image-20210202120003821](https://user-images.githubusercontent.com/38639633/106569231-9decfd80-6577-11eb-8110-9d7807a269d2.png)



## 2.6. Bagging vs Boosting

**Bagging **: **B**ootstrapping **agg**regat**ing** 

- multiple models are being trained with bootstrapping
- ex) Base classifiers are fitted on random subset where individual predictions are aggregated(voting or averaging).

**Boosting** 

- It focuses on those specific training samples that are hard to classify.
- A strong model is built by combining weak learners in sequence where each learner learns from the mistakes of the previous weak learner.



bagging이 데이터 샘플링을 다르게 하는 것이라면, Boosting은 잘맞춘 데이터와 못 맞춘 데이터들을 sequential하게 합쳐 strong한 모델로 만드는 것

![image-20210202121951698](https://user-images.githubusercontent.com/38639633/106569234-9decfd80-6577-11eb-9066-667a2fbdc739.png)



# 3. Practiacal Gradient Descent Methods

실험적 관점에서의 GD(Gradient Descent) 방법들

- Stochastic gradient descent
	- 한 번에 한 개의 데이터만 보게해서 업데이트하고, 이렇게 여러번 반복하는 방법
	- 하나의 sample
- Mini-batch gradient descent
	- batch 단위긴 하지만 전체 데이터의 미니 subset을 뜯어서 학습하는 방법
- batch gradient descent
	- 한번에 전체 데이터를 전부 사용하고 업데이트하는 방식

## 3.1. Batch-size Matters

단순히 자원적 한계를 해결하기 위해 데이터를 split하는 것 이외에 더 큰 의미를 가진다. 

> On Large-batch Training for Deep Learning: Generalization Gap and Sharp Minima, 2017
>
> *"It has been observed in practice that when using a larger batch there is a degradation in the quality of the model, as measured by its ability to generalize"*
>
> *"We ... present numerical evidence that supports the view that large batch methods tend to converge to **sharp minimizers** of the training and testing functions. In contrast, small-batch methods consistently converge to **flat minimizers**... this is due to the inherent noise in he gradient estimation."*
>
> Flat Minimum이 generalization performance가 더 좋다

![image-20210202123123221](https://user-images.githubusercontent.com/38639633/106569235-9e859400-6577-11eb-86bb-41a21a14de2b.png)

## 3.2. Gradient Descent Methods

결국 실제적으로는 Loss function을 정의한 뒤 편미분을 시행해야하는데, 손으로는 너무 어렵고 힘든 과정이므로 automatic한 미분을 해야한다. 

그렇기에 우리는 Gradient Descent 방법을 선택해야한다. 아래

> (Stochastic) Gradient descent
>
> Momentum
>
> Nesterov Accelerate
>
> Adagrad
>
> Adadelta
>
> RMSprop
>
> Aam



### 3.2.1. (Stochastic) Gradient descent

$$
W_{t+1}\leftarrow W_t -\eta g_t
$$

- Gradient descent에는 문제점이 몇 가지 있다. 
	- Learning rate인 $\eta$를 찾기가 힘들다. 너무 작으면 학습이 너무 안되고, 크면 local minima를 지나친다.

<br>



### 3.2.2. Momentum

$$
a_{t+1} \leftarrow \beta a_t + g_t\\
W_{t+1}\leftarrow W_t -\eta a_{t+1}
$$

- 한국말로 말하면 `관성`
- $\beta$라고 불리는 `momentum`이 한번 정해지면
- $a_{t+1}$로 표현되는 `acumulation`, 즉 모멘텀이 포함된 그래디언트로 업데이트 시키는 방법
- 장점:
	- 한번 흘러가면 해당 방향으로 흘러가는 편이기 때문에, gradient $g_t$ 값이 흔들려도 잘 학습되는 편이다. 

<br>



### 3.2.3. Nesterov Accelerate Gradient(NAG)

$$
a_{t+1} \leftarrow \beta a_t + \nabla\mathcal{L}(W_t - \eta\beta a_t)\\
W_{t+1}\leftarrow W_t -\eta a_{t+1}
$$

- 모멘텀은 현재 가지고 있는 파라미터에서 그래디언트를 계산하고, 이를 통해 그래디언트를 accumulation 했다

- NAG에서 $\nabla\mathcal{L}(W_t - \eta\beta a_t)$에 해당하는 `Lookahead gradient`는 한 스텝 더 간 뒤에, 간 곳에의 그래디언트를 통해 accumulation한다. 

	![image-20210202124802429](https://user-images.githubusercontent.com/38639633/106569237-9e859400-6577-11eb-820e-3c724cb5e121.png)

- 장점 : 

	- 모멘텀과 다르게 로컬미니멈을 잘 찾는다. 
	- 미리 한 스텝 더 보고, 만일 local minimum을 지났다면(그래디언트 부호가 바뀌는걸 캐치) 다시 돌아오거나 스텝을 조절

<br>



### 3.2.4. Adagrad

Adagrad **adap**ts the learning rate, performing larger updates for infrequent and smaller updates for frequent parameters.

뉴럴 네트워크의 파라미터가 많이 변한 파라미터에 대해서는 적게변화시키고, 반대로 적게 변화한 파라미터에 대해서는 많이 변화시키는 방식

![image-20210202125403321](https://user-images.githubusercontent.com/38639633/106569239-9f1e2a80-6577-11eb-85de-824b078da0c9.png)

$G_t$ : 파라미터가 얼만큼 변했는지에 대한 값을 저장. 변한 정도를 제곱해서 더한 값. 분모에 들어갔기 때문에,  많이 변할 수록 값이 작아지고, 적게 변할수록 값이 커진다. 

**단점**

- 결국 $G_t$의 값이 커지게 되면 학습을 하면 할 수록 gradient 앞 계수가 작아져서 학습 진행이 안된다. 



<br>



### 3.2.5. Adadelta

**Adadelta** extends **Adagrad** to reduce its monotonically decreasing the learning rate by restricting the accumulation window.

Adagrad가 지닌 단점을 극복하기 위해 등장. adagrad와 달리 전체 gradient의 제곱을 합해서 적용하는 것이 아니라, window를 설정하여 어느 정도의 time step에 따른 범위만큼만 gradient의 제곱을 보겠다는 아이디어 

![image-20210202130321796](https://user-images.githubusercontent.com/38639633/106569241-9f1e2a80-6577-11eb-9e70-1d7d21a41a77.png)

> EMA : Exponential moving average

특징 중 하나는 Learning rate가 없다. 

<br>



### 3.2.6. RMSprop

RMSprop is an unpublished, adaptive learning rate method proposed by Geoff Hinton in his lecture.

논문을 통해서 제안된 것은 아니고, 제프리 힌튼의 강의 중 *<u>"이렇게 해보니 잘된다"</u>*며 했던 말에서 시작되었다.

![image-20210202130800326](https://user-images.githubusercontent.com/38639633/106569243-9fb6c100-6577-11eb-8ddd-81ddba574a33.png)



보이는 것과 같이 Adadelta에서 없던 stepsize를 넣고 $H_t$ term을 뺀 것이 전부이다. 

<br>



### 3.2.7. Aam

**Adaptive Moment Estimation(Adam)** leverages both past gradients and squared gradients.

앞서 공부했던 모멘텀 방식을 합친 방식이다. 

![image-20210202133111854](https://user-images.githubusercontent.com/38639633/106569245-9fb6c100-6577-11eb-8a02-3dd5f86d6753.png)



# 4. Regularization

규제. 방해를 통해 학습을 방해함으로써 해당 방법론이 학습데이터 뿐만아니라 테스트 데이터에서도 좋은 성능을 발휘하게 하기 위함이다. 

> Early stopping
>
> Parameter norm penalty
>
> Data augmentation
>
> Noise robustness
>
> Label smoothing
>
> Dropout
>
> Batch normalization

<br>



## 4.1. Parameter Norm Penalty

파라미터가 너무 커지지 않게 해주는 방법

해석적 의미 : neural network가 만드는  function space 함수의 공간 속에서 이 함수를 최대한 부드러운 함수로 보자는 것이고, 부드러울 수록 generalize performance가 좋을 것이라는 가정을 바탕으로 이뤄진다.

![image-20210202163409817](https://user-images.githubusercontent.com/38639633/106569247-a04f5780-6577-11eb-97e7-8c18530e33ea.png)

<br>



## 4.2. Data Augmentation

아래와 같이 데이터가 작을 때 모델의 성능을 잘못 이해할 수 있다.

![image-20210202163628085](https://user-images.githubusercontent.com/38639633/106569249-a04f5780-6577-11eb-8b02-ab01eeb87b08.png)

이를 해결하기 위해 데이터 augmentation을통해 모델에 대한 generalization performance를 높일 수 있다. 특히, convolution의 경우 이미지 데이터를 flip, rotation 함으로써 모델의 성능을 향상시킬 수 있다. 

<br>



## 4.3. Noise Robustness

실험적 측면에서 노이즈를 넣어줬을 때 학습이 더 잘되는 결과가 있다.

<br>



## 4.4. Label Smoothing

**Mix-up** : Constructs augmented training examples by mixing both input and output of two randomly selected training data.

> 서로 다른 레이블을 지닌 두 데이터의 레이블 값을 50:50으로 섞고, 이미지도 섞는다. 
>
> ![image-20210202164701336](https://user-images.githubusercontent.com/38639633/106569254-a0e7ee00-6577-11eb-8da5-a7334b1b36e5.png)



**Cutout** : 이미지의 일정 영역 자체를 잘라버리는 방법

**CutMix** : Constructs augmented training examples by mixing inputs with cut and paste and outputs with soft labels of two randomly selected raining data.

> 믹스업과 다르게, 이미지 자체를 블렌딩하는 것이 아니라, 영역 자체를 비율로 잘라서 구분해준다. 
>
> ![image-20210202164727908](https://user-images.githubusercontent.com/38639633/106569256-a1808480-6577-11eb-84ec-4613cbcb35fd.png)

