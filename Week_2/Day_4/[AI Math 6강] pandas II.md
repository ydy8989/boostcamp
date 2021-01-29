# [AI Math 6강] pandas II

 pandas I 강의에 이어서 pandas 라이브러리의 다음과 같은 기능에 대해 알아봅니다.

- groupby
- pivot_table
- joint method (merge / concat)
- Database connection
- Xls persistence 

> 강의에서 사용 될 예제 코드는 **[여기](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/tree/main/codes/pandas/part_2)**에서 확인 가능합니다. 

<br>

# 1. groupby

- 묶음의 기준이 되는 컬럼에 적용받는 컬럼을 어떻게할지

- ```python
  df.groupby('team')['Point'].sum()
  ```

  > ![image-20210128155107847](https://user-images.githubusercontent.com/38639633/106174269-2c5c2a80-61d8-11eb-98d5-1110d4e41d8f.png){: width="45%"}{: .center}![image-20210128155113720](https://user-images.githubusercontent.com/38639633/106176866-2c115e80-61db-11eb-98ff-ad7c5d963a33.png){: width="45%"}{: .center}

- 한 개 이상의 Column을 묶을 수도 있다. 

  ```python
  df.groupby(['Team','Year'])['Point'].sum()
  ```

  > ![image-20210128155450884](https://user-images.githubusercontent.com/38639633/106174324-43028180-61d8-11eb-830a-3b488ecd778c.png)

<br>

## 1.1. Hierarchical index

- Goupby 명령의 결과물도 결국에는 DataFrame이다. 

- 두 개의 column으로 groupby를 할 경우, index가 두 개 생성된다.

  > ![image-20210128155947558](https://user-images.githubusercontent.com/38639633/106174349-48f86280-61d8-11eb-8b75-80465ea0ea56.png)

- **unstack()**

  - Group으로 묶여진 데이터를 matrix 형태로 전환해준다. 

    > ![image-20210128160055110](https://user-images.githubusercontent.com/38639633/106174370-501f7080-61d8-11eb-8dc8-fb42d71718b0.png)

- **swaplevel()**

  - Index level을 변경할 수 있다. 
    - Multi index의 경우 `index1, index2` => `index2, index1` 의 순서로 변경

- **operations**

  - Index level을 기준으로 기본 연산수행이 가능하다. 

    ```python
    h_index.sum(level = 0) # 1번째 인덱스를 기준으로 summation
    h_index.sum(level = 1) # 2번째 인덱스를 기준으로 summation
    ```

<br>

## 1.2. grouped

- Groupby에 의해 Split된 상태를 추출할 수 있다. 

  ```python
  grouped = df.groupby('Team')
  for name, group in grouped:
      print(name)
      print(group)
  ```

  > ![image-20210128185034495](https://user-images.githubusercontent.com/38639633/106174424-5f062300-61d8-11eb-9e77-0595e6581b74.png)

- 특정 key값을 가진 그룹의 정보만 추출 가능하다.

  ```python
  grouped.get_group("Devils")
  ```

  > - `.get_group()`로 해당 group에 대한 정보만 추출한 모습
  >
  > ![image-20210128185056706](https://user-images.githubusercontent.com/38639633/106174443-64636d80-61d8-11eb-9a79-bcbf98cacf7b.png)

- 추출된 group 정보에는 세 가지 유형의 apply가 가능하다.

  > 1. Aggregation : 요약된 통계 정보를 추출해줌
  > 2. Transformation :  해당 정보를 변환해줌
  > 3. Filtration : 특정 정보를 제거하여 보여주는 필터링 기능

<br>



### Aggregation(.agg(연산명))

- ```python
  grouped.agg(max)
  ```

  > ![image-20210128185804544](https://user-images.githubusercontent.com/38639633/106174486-70e7c600-61d8-11eb-9fed-40b9188df7cf.png)

- ```python
  import numpy as np
  grouped['Points'].agg([np.sum, np.mean, np.std])
  ```

  > 하나의 특정 컬럼에 여러개의 function을 apply 할 수도 있다. 
  >
  > ![image-20210128190047309](https://user-images.githubusercontent.com/38639633/106174524-79d89780-61d8-11eb-9721-2c84c5c7af1c.png)

  <br>

### Transofrmation

- Aggregation과 달리 Key값 별로 요약된 정보가 아님

- 개별 데이터의 변환을 지원한다. 

- $$
  z_i=\frac{x_i-\mu}{\sigma}
  $$

  > ```python
  > # score로 표준화 함수를 정의한 뒤 transform
  > score = lambda x: (x - x.mean()) / x.std()
  > grouped.transform(score)
  > ```
  >
  > ![image-20210128190503913](https://user-images.githubusercontent.com/38639633/106174558-83fa9600-61d8-11eb-9308-9d7ae00a8c17.png)

### Filter

- 특정 조건으로 데이터를 검색할 때 사용한다. 

  ```python
  df.groupby("Team").filter(lambda x: len(x) >= 3)
  ```

  > filter 안에는 boolean 조건이 존재해야한다.
  >
  > len(x)는 grouped된 dataframe의 개수를 의미한다. 
  >
  > ![image-20210128191447560](https://user-images.githubusercontent.com/38639633/106174580-89f07700-61d8-11eb-92db-2536d3009fe9.png)

<br>

# 2. Case study

> 해당 섹션은 [예제](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/pandas/part_2/1_groupby_hierarchical_index.ipynb)로 대체한다. 

<br>

# 3. Pivot table

- excel에서의 피벗테이블과 같다. 

- index축은 groupby와 동일하다

- column에 추가로 labeling 값을 추가하여 value에 numeric type 값을 aggregation하는 형태

  ```python
  df_phone = pd.read_csv("./data/phone_data.csv")
  df_phone["date"] = df_phone["date"].apply(dateutil.parser.parse, dayfirst=True)
  df_phone.head()
  ```

  > ![image-20210128192552712](https://user-images.githubusercontent.com/38639633/106174596-8f4dc180-61d8-11eb-9437-6559dcfa9967.png)

  ```python
  df_phone.pivot_table(
      values=["duration"],
      index=[df_phone.month, df_phone.item],
      columns=df_phone.network,
      aggfunc="sum",
      fill_value=0,
  )
  ```

  > ![image-20210128192623488](https://user-images.githubusercontent.com/38639633/106174618-94127580-61d8-11eb-8d8f-1bcbc8bd6abc.png)

  <br>

# 4. Crosstab

- 특히 두 column의 교차 빈도, 비율, 덧셈 등을 구할 때 사용한다. 

- Pivot table의 특수한 형태

- User-Item Rating Matrix 등을 만들 때 사용가능함

  ```python
  df_movie = pd.read_csv("data/movie_rating.csv")
  df_movie.head()
  ```

  > ![image-20210128192828354](https://user-images.githubusercontent.com/38639633/106174639-9a085680-61d8-11eb-896a-3e854feb1e5f.png)

  ```python
  pd.crosstab(
      index=df_movie.critic,
      columns=df_movie.title,
      values=df_movie.rating,
      aggfunc="first",
  ).fillna(0)
  ```

  > ![image-20210128192906252](https://user-images.githubusercontent.com/38639633/106174665-a096ce00-61d8-11eb-8e6c-7105cded1303.png)




<br>



# 5. Merge & Concat

- SQL에서 많이 사용하는 Merge와 같은 기능 
- 두 개의 데이터를 하나로 합침



<br>



## 5.1. pd.merge -  `on = `

- `pd.merge(df_a, df_b, on = '컬럼명')`

  > on : 공통적으로 있는 컬럼에 대한 merge 진행

- `pd.merge(df_a, df_b, left_on = '좌측df컬럼명', right_on = '우측df컬럼명')`

  - 두 df의 column명이 다를 때 사용한다.

<br>

## 5.2. join method - `how = `

- join method는 총 4가지 방식이 있다. 

  ![image-20210128203251180](https://user-images.githubusercontent.com/38639633/106174692-a7254580-61d8-11eb-87c6-ca702154dd88.png)

  > 더 자세한 코드 예제는 [여기](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/pandas/part_2/3_merge_concat.ipynb)

- **Index based join**

  컬럼명이 아닌 index를 기준으로 merge할 때 사용한다. 

  이 경우, 동일 컬럼명이 존재할 시에 `_x`, `_y`가 컬럼명 뒤에 붙는다. (주의)

  ```python
  pd.merge(df-a, df_b, right_index = True, left_index = True)
  ```

  > ![image-20210128203649741](https://user-images.githubusercontent.com/38639633/106174706-ad1b2680-61d8-11eb-9499-e9bd12574e83.png)

  

<br>



## 5.3. Concat

- 같은 형태의 데이터(DataFrame)를 붙이는 연산작업

  - `merge`와 다른 점은 전체를 붙인다는 점에 있다.

  ```python
  df_new = pd.concat([df_a, df_b], axis = 1)
  ```

  >  axis 설정을 통해 위아래로 붙일 건지, 좌우로 붙일 건지를 선택한다. 

  <br>

# 6. Persistence

## 6.1. Database connection

- Data loading시에 db connection 기능을 제공한다. 

  ```python
  import sqlite3  # pymysql <- 설치
  
  conn = sqlite3.connect("./data/flights.db")
  cur = conn.cursor()
  cur.execute("select * from airlines limit 5;")
  results = cur.fetchall()
  results
  ```

## 6.2. XLS persistence
- Dataframe의 엑셀 추출코드

- Xls 엔진으로 openpyxls 또는 XlsxWrite 사용

- Install

  ```python
  ### Pandas persistence
  #### install
  - conda install openpyxl
  - conda install XlsxWriter
  - see more http://xlsxwriter.readthedocs.io/working_with_pandas.html
  ```

  ```python
  import pandas as pd
  writer = pd.ExcelWriter("./df_routes.xlsx", engine="xlsxwriter")
  # writer
  df_routes.to_excel(writer, sheet_name="Sheet1")
  writer.close()
  ```

  > **주의!**
  >
  > 마지막 객체를 닫아주는 .close()를 사용해야 파일이 생성된다. 

## 6.3. Pickle persistence

- 가장 일반적인 python 파일 persistence

- to_picle, read_pickle함수를 사용한다.

  ```python
  df_routes.to_pickle("./data/df_routes.pickle")
  df_routes_pickle = pd.read_pickle("./data/df_routes.pickle")
  df_routes_pickle.head()
  ```