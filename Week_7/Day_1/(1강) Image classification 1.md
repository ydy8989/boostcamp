# (1강) Image classification 1

**강의 소개**

우리는 오감 중 특히 시각에 의존하여 사물을 바라보고 이해하며 살아가고 있습니다.
동일한 프로세스를 컴퓨터에 적용한 컴퓨터 비전입니다.

본 강의에서는 컴퓨터 비전 (CV)의 첫 시간으로 CV에 대해 짧게 소개하고, CV에서 가장 기본적인 task, image clasiification을 소개합니다. Image Classification은 사진이 주어졌을 때 특정 카테고리로 분류하는 task입니다.

이번 강의에서는 먼저 기존의 머신러닝과 구분되는 딥러닝을 사용한 Image classification의 특징에 대해서 배웁니다. 다음으로 대표적인 CNN 모델인 AlexNet을 배우고 이에 대한 실습을 진행합니다.
끝으로 가장 유명한 classification 모델 중 하나인 VGGNet에 대해 배웁니다.

**Further Reading**

\- VGGNet : https://arxiv.org/pdf/1409.1556.pdf





## Course overview

### Why is visual perception important?

**Artificial Intelligence(AI)?**

> The theory and development of computer systems able to perform tasks normally requiring human intelligence, such as visual perception, speech recognition, decision- making, and translation between languages.
>
> -from the Oxford dictionary

- 사고하고, 인과관계 분석 이외에도 시각, 소리에 대한 인지, 그리고 이해하는 것도 사람의 지능에 포함시키는 사전적 의미가 정의된다. 



**Humans learn about the world through multi-modal perception**

인공지능의 레퍼런스를 찾아본다면 인간이다. 

- 지능을 구현하기 위해서 가장 첫 번째 스텝으로는 지각능력의 획득이 중요하다. 
- 갓난 아이가 맛보고, 듣고, 만지며 배우듯



**perception to system?**

- It's(input, output) data
- 인공지능의 입력과 출력이 사람이 이해하기 쉬울수록 좋다. 
- 사람은 오감을 이용해 상호작용하면서 학습한다. 
- 하지만 교차감각, 다중 감각 등을 이용해 더 복잡한 감각으로 정보를 획득한다. 
	- 더 나아가 social perception. 즉, 사회적 경험 및 악수, 말투에서 오는 설득력과 같이 복합 감각으로부터 학습하게 된다.
- **developing machine perception is still an open research area**

 ![image](https://user-images.githubusercontent.com/38639633/110263601-91d2e080-7ffa-11eb-9893-4d85df954bd5.png)



- 인간의 정보중 많은 부분을 차지하는 것은 시각적 정보가 크다. 



### What is computer vision?

그럼 컴퓨터 비전과 사람의 시각이 어떤 연관성이 있을까?

- 사람이 눈으로 장면을 관찰한다(sensing)

- 수정체 뒤에 상이 맺힌다. (sensing)

- 뇌에 자극을 전달하고 이해한다. 

- 뇌에서 해석하고, 이해한다. 

	---

- 카메라로 찍는다.

- GPUs에 올리고 알고리즘을 통해 연산한다

- 장면을 통한 분석이 자료구조로 출력된다. (interpretation or representation)

- `+알파`이 출력으로 원래 이미지를 구현하거나 복원하는 작업을 Computer graphics(rendering)이라고 한다.

	- 이 rendering을 반대로 하는 작업을 Computer viosion(혹은 inverse rendering)이라고 한다.

![image](https://user-images.githubusercontent.com/38639633/110263951-7ddbae80-7ffb-11eb-94cd-f235b9e94ef0.png)



그럼 컴퓨터에게 세상을 바라보는 눈을 어떻게 만들어 줘야할까?

- Visual perception & interlligence
	- input : visual data
- Class of visual perception
	- Color perception
	- Motion perception
	- 3D perception
	- Semantic-level perception
	- Social perception (emotion perception)
	- Visuomotor perception, etc.
- 이런 것들을 구현하는 것 뿐만 아니라, 사람의 biological한 것을 이해하고, 이를 어떤 알고리즘으로 구현할지에 대한 것들 또한 포함한다. 



사람의 시각 능력은 불완전하다. 

- To develop machine visual perception,

	- We need to understand the good and bad of our visual perception
	- We need to come up with how to compensate for the imperfection

	![image](https://user-images.githubusercontent.com/38639633/110266929-2ab92a00-8002-11eb-9969-3dbf5686bcb3.png)

- 우리 인간의 눈은 위와 같은 기괴한 얼굴에 대한 데이터가 없기에 어느정도 biased 되어 학습되었다고 생각할 수 있다. 

- 이는 거꾸로 된 이미지만을 봤을 때, 이상함을 쉽게 눈치채지 못하는 이유중 하나이다. 



**어떻게 하면 컴퓨터를 활용해서 시각 지각능력을 학습할까?**

**machine learning**

- input data
- 전문가의 feature extraction (사람)
- 간단한 분류기 등을 적용
- output 출력



**Deep learning**

- input 
- feature extraction + classification
- output



### What you will learn in this course

- Fundamental image tasks
- multi-modal learning( vision + {text, sound, 3d, etc..})
- conditional generative model
- Neural network analysis by visualization



## Image classification

