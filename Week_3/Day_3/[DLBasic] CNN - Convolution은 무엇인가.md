#  [DLBasic]What is Convolution?

CNN(Convolutional Neural Network)에서 가장 중요한 연산은 **Convolution** 입니다. CNN에 대한 공부를 하기 전에 **Convolution의 정의, convolution 연산 방법과 기능**에 대해 배웁니다. 그리고 Convolution, 입력을 축소하는 Pooling layer, 모든 노드를 연결하여 최종적인 결과를 만드는 Fully connected layer 로 구성되는 **기본적인 CNN(Convolutional Neural Network) 구조**에 대해 배웁니다. 



## 2.1. Convolution

필터의 모양에 따라서 같은 이미지가 다르게 표현될 수 있다. 

![conv](../../assets/img/boostcamp/conv.gif)
$$
\begin{align}
O_{11} = &I_{11}K_{11}+I_{12}K_{12}+I_{13}K_{13}+I_{21}K_{21}+I_{22}K_{22}+I_{23}K_{23}+I_{31}K_{31}+I_{32}K_{32}\\
&+I_{33}K_{33}+bias\\
O_{12} = &I_{12}K_{11}+I_{13}K_{12}+I_{14}K_{13}+I_{22}K_{21}+I_{23}K_{22}+I_{24}K_{23}+I_{32}K_{31}+I_{33}K_{32}\\
&+I_{33}K_{33}+bias\\
O_{13} = &I_{13}K_{11}+I_{14}K_{12}+I_{15}K_{13}+I_{23}K_{21}+I_{24}K_{22}+I_{25}K_{23}+I_{33}K_{31}+I_{34}K_{32}\\
&+I_{34}K_{33}+bias\\
O_{14} = &I_{14}K_{11}+I_{15}K_{12}+I_{16}K_{13}+I_{24}K_{21}+I_{25}K_{22}+I_{26}K_{23}+I_{34}K_{31}+I_{35}K_{32}\\
&+I_{35}K_{33}+bias\\
\end{align}
$$


<br>



## 2.2. CNN

![image-20210203225547769](../../assets/img/boostcamp/image-20210203225547769-1612361470642.png)

CNN consists of convolution layer, pooling layer, and fully connected layer.

- Convolution and pooling layers: feature extraction
- Fully connected layer : decision making (e.g., classification)

![image-20210203230804717](../../assets/img/boostcamp/image-20210203230804717-1612361470642.png)

> 필수적이진 않지만, stride와 channel을 보고 파라미터의 크기를 가늠할 수 있어야 한다.

*stride와 padding에 대한 설명은 생략하도록 한다.*

<br>



## 2.3. Convolution Arithmetic

>  Padding (1), Stride (1), $3\times 3$ Kernel

![image-20210203231923207](../../assets/img/boostcamp/image-20210203231923207.png)

결국 Input의 Width와 Height는 convolution 연산을 수행하는 과정에서 발생하는 parameter 수에 상관이 없다. 

Kernel size 3$\times$3은 사실 3$\times$3$\times$`Channel size`이므로 (그림의 파란색 부분) 파라미터는 3$\times$3$\times$128(이전 channel size)$\times$64(변한 뒤의 channel size)가 된다. 

