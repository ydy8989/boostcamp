# (4강) Data Generation

**강의 소개**

Vanilla 데이터를 가지고 Dataset을 구성한 다음 모델에 빠르고, 효율적으로 Feeding 하기 위해 알아야 할 것들에 대해서 다룹니다. 강의에서 Data Feeding이라고 말하는 것의 개념과, 실제로 이것을 제대로 하지 않았을 때 어떤 일들이 일어날 수 있는지 알아보겠습니다.



그리고, 파이토치에서 torch.utils.data에 있는 Dataset, DataLoader에 대한 설명과, 그 차이를 다뤄보겠습니다.



## Data Feeding

### Feed : 먹이를 주다

먹이를 주다 = 대상의 상태를 고려해서 적정한 양을 준다.

![image](https://user-images.githubusercontent.com/38639633/112923643-9ac25800-9149-11eb-979f-2688a982952e.png)

- 예전 tensorflow 버전1의 경우 placeholder를 만들고 feed_dict를 통해 데이터를 입력(?)했다. 
- 데이터를 feed한다는 의미를 곱씹자.



**모델에 먹이(Data)를 주다?**

![image](https://user-images.githubusercontent.com/38639633/112925037-29d06f80-914c-11eb-9200-d83b744f0855.png)- 

- 모델의 역량과 데이터 제너레이터의 역량을 잘 파악하고 있어야만 한다. 



### 데이터의 전처리 순서에 따른 속도

![image](https://user-images.githubusercontent.com/38639633/112925194-7025ce80-914c-11eb-8b4f-ddf5529c1a8d.png)

- 리사이즈의 위치에 따라 속도가 달라졌음을 알 수 있다. 
- 리사이즈를 돌리고 로테이션을 돌리면 더 많은 시간이 소모된다. 
	- 튜닝의 한 방식이다. 
	- 다양한 테스트를 통해 성능 개선을 시도하자.





## torch.utils.data

Dataset의 구조

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self): 
        '''
        MyDataset 클래스가 처음 선언 되었을 때 호출
        '''
        pass
    
    def __getitem__(self, index):
        '''
        MyDataset의 데이터 중 index위치의 아이템을 리턴
        - 실질적으로 파일을 읽어서 array를 읽어서 뱉어주는 로직이 필요
        '''
        return None
    
    def __len__(self):
        '''
        MyDataset의 아이템 전체의 길이
        '''
        return None
```



### DataLoader

내가 만든 Dataset을 효율적으로 사용할 수 있도록 관련 기능을 추가한다.

```python
train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size = batch_size, 
                                           num_workers=num_workers, drop_last = True)
```



만든 데이터 셋을 넣었을 때, (Batch, Channel, Height, Width)를 뱉도록 한다. 

![image](https://user-images.githubusercontent.com/38639633/112927781-7fa71680-9150-11eb-8389-502effadbe11.png)

- 다양한 기능이 많다. 
- 추가적인 내용은 [https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)에서 확인하자.
- 재사용의 관점에서 Dataset과 DataLoader는 분리해서 정의하고, 관리하는 것이 좋다. 







