**CBOW(Continuous Bag-of-Words)**

- 주변 단어들을 가지고 중심 단어를 예측하는 방식으로 학습합니다.

- 주변 단어들의 one-hot encoding 벡터를 각각 embedding layer에 projection하여 각각의 embedding 벡터를 얻고 이 embedding들을 element-wise한 덧셈으로 합친 뒤, 다시 linear transformation하여 예측하고자 하는 중심 단어의 one-hot encoding 벡터와 같은 사이즈의 벡터로 만든 뒤, 중심 단어의 one-hot encoding 벡터와의 loss를 계산합니다.

- 예)

	 

	A cute puppy is walking

	 

	in the park. & window size: 2

	- Input(주변 단어): "A", "cute", "is", "walking"
	- Output(중심 단어): "puppy"

**Skip-gram**

- 중심 단어를 가지고 주변 단어들을 예측하는 방식으로 학습합니다.

- 중심 단어의 one-hot encoding 벡터를 embedding layer에 projection하여 해당 단어의 embedding 벡터를 얻고 이 벡터를 다시 linear transformation하여 예측하고자 하는 각각의 주변 단어들과의 one-hot encoding 벡터와 같은 사이즈의 벡터로 만든 뒤, 그 주변 단어들의 one-hot encoding 벡터와의 loss를 각각 계산합니다.

- 예) 

	A cute puppy is walking

	 in the park. & window size: 2

	- Input(중심 단어): "puppy"
	- Output(주변 단어): "A", "cute", "is", "walking"