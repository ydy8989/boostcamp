**[3/29] Day 1 - P Stage Start !**

학습목표

- P stage 목적의 이해
- Competition의 이해
- Image 데이터의 이해와, EDA 본질 이해
- 베이스라인 제출 프로세스 이해

**1강 [ Competition with AI Stages! ]**

1. Welcome, P Stage !
	- P stage 시작!
	- U stage와 P stage의 차이
2. Competition with AI Stages!
	- Competition이란?
	- AI Stages
	- Competition Detail
		- Competition의 각 구성요소의 의의
		- Overview, Data description, Notebook, Submission, Discussion
	- 왜 P Stage는 Competition을 선택했나?
		- Machine Learning Pipieline

**2강 [ Image Classification & EDA ]**

1. Overview
2. EDA
	- EDA = Exploratory Data Analysis : 탐색적 데이터 분석
	- EDA에서 뭘 해야할 지 모르겠어요..?
	- 사실 EDA의 진짜 목적은..
3. Image + Classification
	- Image?
	- Input + Model = Output
	- Image Classification = Image + Classification Model = Class
4. Baseline
	- 베이스라인 코드를 제공
	- 하기에 따라 상당히 많은 부분을 배워갈 수 있음

**[3/30] Day 2 - Data Feeding**

학습목표

- Vanilla Data → Dataset
- Pre-processing 이해
- Augmentation 이해
- Data Feeding 의미 제대로 이해하기

**3강 [ Dataset ]**

1. Overview
2. Pre-processing
	- Data Science = 0.8 * pre-processing + 0.2 * etc.
3. Generalization
	- Bias & Variance
	- Train / Validation
	- Data Augmentation
		- torchvision
		- albumentations
4. 무조건이라는 것은 없다.

**4강 [ Data Generation ]**

1. Overview
2. Data Feeding
	- 모델에 먹이를 주다?
	- 모델에 먹이(Data)를 잘 주는 방법.
3. torch.utils.data
	- Dataset
	- DataLoader
	- Dataset과 DataLoader는 다른 것입니다.

**[3/31] Day 3 - Model**

 학습목표

- Image Classification 모델 구조에 대해 이해하기
- Pytorch 모델 구현 코드 이해하기

**5강 [ Model 1 ]**

1. Overview
2. Design Model with Pytorch
	- Pytorch
	- nn.Module
	- forward
	- Parameter

**6강 [ Model 2 ]**

1. Overview
	- Computer Vision
	- ImageNet
2. Pretrained Model
	- 배경
	- torchvision.models
3. Transfer Learning
	- CNN base의 모델 구조 (simple)
	- 내 데이터, 모델과의 유사성
	- Case by Case
		- Feature Extraction
		- Fine tuning

**[4/1] Day 4 - Training & Inference**

 학습목표

- Pytorch로 학습, 추론과정을 구현한 코드를 이해
- Loss, Optimizer, Metric의 이해
- Pytorch 에서 Loss, Optimizer, Metric 표현 방식을 이해

**7강 [ Training & Inference 1 ]**

1. Overview
2. Loss
	- 복습: (오차) 역전파
	- Loss도 사실은 nn.Module
	- loss.backward()
	- 조금 특별한 loss
3. Optimizer
	- 어느 방향으로 얼마나 움직일지?
	- LR scheduler
		- StepLR
		- CosineAnnealingLR
		- ReduceLROnPlateau
4. Metric
	- 모델의 평가
	- Metric의 허와 실
	- 올바른 Metric의 선택

**8강 [ Training & Inference 2 ]**

1. Overview
2. Training Process
	- Training Process 이해
	- More: Gradient Accumulation
3. Inference Process
	- Inference Process 이해
	- Validation 확인
	- Checkpoint
	- 최종 Submission
4. Appendix: Pytorch Lightning

**[4/2] Day 5 - More..**

 학습 목표

- 앙상블 개념 이해
- Cross Validation 구조 이해
- Experiment Visualization

**9강 [ Ensemble ]**

1. Overview
2. Ensemble
	- Model Averaging (Voting)
	- Cross Validation
		- Stratified K-fold
	- TTA (Test Time Augmentation)
3. Hyperparameter Optimization

 **10강 [ Experiment Toolkits & Tips ]**

1. Overview
2. Training Visualization
	- Tensorboard
	- wandb
3. Machine Learning Project
	- Jupyter Notebook
	- Python IDLE
4. Some Tips