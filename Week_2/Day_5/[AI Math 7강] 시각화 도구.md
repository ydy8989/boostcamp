# [AI Math 7강] 시각화 도구

데이터 분석을 위해서 가장 기초적인 단계로 해당 데이터를 시각화하여 보여줄 수 있는 능력이 필요합니다. 시각화 라이브러리는 파이썬에서만 10개가 넘는등 다양한 라이브러리가 존재합니다. 우리는 그 중에서 가장 대중적으로 많이 쓰이는 **matplotlib, seaborn** 모듈에 대해 학습합니다

**matplotlib**

가장 대중적으로 많이 쓰였고 많은 파이썬 라이브러리의 근간이 되는 matplotlib 입니다. matplotlib는 다른 라이브러리들의 부모 라이브러리로서의 역할을 하고 있다고 표현할정도로 다른 라이브러리들에 많은 영향을 주었습니다. 다소 복잡한 라이브러리 구성으로 인해 최근에는 그 사용 빈도와 대중성이 떨어지고 있으나 여전히 많은 입문자들이 처음 사용해보게 되는 좋은 시각화 라이브러리입니다.

**seaborn**

시각화를 위한 다양한 기능을 손쉽게 사용할 수 있도록 지원합니다. 모든 기능은 matplotlib을 기반으로 제공되어 matplotlib과 상호 호환됩니다. matplotlib의 모든 기능을 사용하면서 손 쉽게 사용하고 싶다면 seaborn이 가장 적절한 대안입니다.



**<u>이번 내용의 코드는 [이곳](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/tree/main/codes/viz)에서 확인할 수 있습니다.</u>**



# 1. Matplotlib

- 개인적으로 matplotlib의 경우에는 많은 사용을 해왔었고, 숙련도가 높다고 생각하기에 따로 정리를 하지는 않겠습니다.
- 위 링크의 예제코드 [`1_1_basic_plot.ipynb`](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/viz/1_1_basic_plot.ipynb), [`1_basic_plot.ipynb`](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/viz/1_basic_plot.ipynb), [`2_data_plot.ipynb`](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/viz/2_data_plot.ipynb)를 통해 각자 코드 리뷰를 진행하는 것을 추천한다. 



# 2. Seaborn

seaborn의 예제코드는 [`3_seaborn.ipynb`](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/viz/3_seaborn.ipynb)에서 확인할 수 있습니다. 

- 기존 matplotlib에 기본 설정을 추가
- 복잡한 그래프를 간단하게 만들 수 있는 wrapper
- 간단한 코드 + 예쁜 결과

### basic plots

- matplotlib과 같은 기본적인 plot

- 손쉬운 설정으로 데이터를 산출할 수 있다. 

- lineplot, scatterplot, countplot 등

  ```python
  sns.lineplot(x='total_bill', y = 'tip', data = tips)
  ```

  > ![image-20210129124334704](https://user-images.githubusercontent.com/38639633/106301490-1f543f80-629b-11eb-82dc-e02d7b04c1bd.png)

  - lineplot의 경우 `data`에 데이터프레임을 넣고, `x`축과 `y`축에 해당하는 column을 넣어 사용한다. 
  - 보이는 것과 같이 데이터의 평균을 main 라인으로 그려주고 분포의 퍼짐 정도를 보여준다.

  

  ```python
  sns.lineplot(x='timepoint', y = 'signal', hue = 'event', data = fmri)
  ```

  > ![image-20210129172820410](https://user-images.githubusercontent.com/38639633/106301488-1ebba900-629b-11eb-9aa6-d64caf9f85b7.png)

  - `hue` : x와 y에 numeric 데이터 컬럼을 넣고, hue에 categorical 데이터를 넣으면, 각 카테고리별 x와 y의 관계를 표현해준다. 

  ```python
  sns.scatterplot(x="total_bill", y="tip", data=tips)
  ```

  > ![image-20210129173042234](https://user-images.githubusercontent.com/38639633/106301487-1ebba900-629b-11eb-9b8b-bff7b8f9458f.png)

  ```python
  sns.regplot(x="total_bill", y="tip", data=tips)
  ```

  > ![image-20210129173137961](https://user-images.githubusercontent.com/38639633/106301485-1e231280-629b-11eb-998a-77ff3c86bbaa.png)

  - `scatterplot`은 말 그대로 점으로 두 column의 관계를 표시해준다. 
  - `regplot`은 scatter에 추가적으로 회귀 선을 그려주는 plotting 메소드이다. 
    - lineplot과 마찬가지로 hue option 사용 가능

  ```python
  sns.countplot(x="smoker", hue="time", data=tips)
  ```

  > ![image-20210129173405640](https://user-images.githubusercontent.com/38639633/106301482-1e231280-629b-11eb-8b7e-889ffcf440dd.png)

  ```python
  sns.set(style="darkgrid")
  sns.distplot(tips['total_bill'])
  ```

  >![image-20210129173513532](https://user-images.githubusercontent.com/38639633/106301479-1d8a7c00-629b-11eb-80a2-f2542ccd6735.png)

### predefined plots

- `Viloinplot` : boxplot에 distribution을 함께 표현
- `Stripplot` : scatter와 category 정보를 함께 표현
- `Swarmplot` : 분포와 함께 scatter를 함께 표현
- `Pointplot` : category별로 numeric의 평균, 신뢰구간 표시
- `regplot` : scatter + 선형함수를 함께 표시



### multiple plots

- 한 개 이상의 도표를 하나의 plot에 작성

- Axes를 사용해서 grid를 나누는 방법

  ```python
  import numpy as np
  import seaborn as sns
  import matplolib.pyplot as plt
  
  sns.set(stype = 'while', palette = 'muted', color_codes=True)
  rs = np.random.RandomState(10)
  
  f, axes = plt.subplots(2, 2, figsize=(7,7), sharex=True)
  sns.despine(left=True)
  
  d = rs.normal(size=100)
  
  sns.distplot(d, ked = False, color = 'b', ax=axes[0,0])
  sns.distplot(d, hist= False, rug = True, color = 'r', ax=axes[0,1])
  sns.distplot(d, hist = False, color = 'g', kde_kws={'shade':True}, ax=axes[1,0])
  sns.distplot(d, color = 'm', ax=axes[1,1])
  
  plt.setp(axes, yticks=[])
  plt.tight_layout()
  ```

  > ![image-20210129174256632](https://user-images.githubusercontent.com/38639633/106301474-1cf1e580-629b-11eb-902a-2584cef8491a.png)



### predefined multiple plots

- `replot` : numeric 데이터 중심의 분포 / 선형표시
- `catplot` : category 데이터 중심의 표시
- `FacetGrid` : 특정 조건에 따른 다양한 plot을 grid로 표시
- `pairplo` : 데이터간의 상관관계 표시
- `Implot` : regression 모델과 category 데이터를 함께 표시

