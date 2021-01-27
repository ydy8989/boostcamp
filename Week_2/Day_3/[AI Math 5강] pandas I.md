# [AI Math 5강] pandas I

**pan**el **da**ta 의 줄임말인 **pandas**는 파이썬의 데이터 처리의 사실상의 표준인 라이브러리입니다.

**pandas**는 파이썬에서 일종의 엑셀과 같은 역할을 하여 데이터를 전처리하거나 통계 처리시 많이 활용하는 피봇 테이블 등의 기능을 사용할 때 쓸 수 있습니다. pandas 역시 numpy를 기반으로 하여 개발되어 있으며, R의 데이터 처리 기법을 참고하여 많은 함수가 구성되어 있거 기존 R 사용자들도 쉽게 해당 모듈을 사용할 수 있도록 지원하고 있습니다.



# 0. Pandas overview

- 구조화된 데이터의 처리를 지원하는 파이썬 라이브러리
- panel data의 약자
- 고성능 array 계산 라이브러리인 numpy와 통합하여, 강력한 '스프레드시트' 처리 기능을 제공한다. 
- 인덱싱, 연산용 함수, 전처리 함수 등을 제공함
- 데이터 처리 및 통계 분석을 위해 사용



## 0.1. 데이터 로딩

- `pd.read_csv`를 이용해 데이터를 로드할 수 있다.
  - url 형식으로 웹의 데이터를 로드할 수 있다. 
  - `sep`옵션 : txt 형식의 데이터를 나눠주는 split 요소를 설정함으로써 데이터를 분리시켜줌
    - 몰랐던 옵션 : `\s+` - 불규칙하게 연속된 single space를 분리해줌. 



# 1. Series

- DataFrame : Data Table 전체를 포함하는 object의 이름

- Series : 데이터프레임 중 하나의 column에 해당하는 데이터의 모음 object를 말한다. 

- column vector를 표현하는 object

- List to series

  - ```python
    list = [1,2,3,4,5]
    example_series = Series(data = list)
    ```

- Dict to series

  - ```python
    dict = {'a':1, 'b':2, 'c':3, 'd':4}
    example_series = Series(data = dict)
    ```

- **[예제 코드](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/pandas/%231/3_pandas_series.ipynb)**

-  인덱스에 접근할 때는 리스트의 인덱스 및 array의 인덱싱과 비슷하다. 



# 2. DataFrame

- 메모리 구조는 index와 columns를 통해 접근하는 방식으로 이뤄져있다. 
- 보통은 `pd.DataFrame`보다는 `pd.read_csv`를 통해 한번에 데이터 프레임을 로드한다. 

## 2.1. DataFrame 생성

- ```python
  # Example from - https://chrisalbon.com/python/pandas_map_values_to_values.html
  raw_data = {
      "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
      "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
      "age": [42, 52, 36, 24, 73],
      "city": ["San Francisco", "Baltimore", "Miami", "Douglas", "Boston"],
  }
  df = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "city"])
  df
  ```

- ![image-20210127173704440](C:\Users\doyeon\AppData\Roaming\Typora\typora-user-images\image-20210127173704440.png)

- 데이터프레임이 갖고 있지 않은 index 및 column명을 호출해주면 그 값들이 `NaN`으로 채워진다. 

## 2.2. DataFrame indexing

- 해당 값에 접근하는 방식은 두 가지가 있다.
  - `df.loc['인덱스명']` : location이라는 의미로, 인덱스의 위치 즉 몇 번째인지를 의미하는 것이 아니라, 인덱스의 이름 그 자체를 찾는다.
  - `df.iloc[인덱스 넘버]` : index location이라는 의미. 인덱스의 이름과 상관없이 몇 번째인지를 찾는다. 
- **[예제 코드](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/pandas/%231/4_pandas_dataframe.ipynb)**



## 2.3. DataFrame handling

- 새 Series에 boolean을 통한 데이터 값을 새로 할당할 수 있다.

  ```python
  df.debt=df.age > 40
  ```

  > df.age의 값이 40 초과면 True, 아니면 False를 df.debt에 새로 입력

- `T` : 데이터프레임 transpose
- `to_csv` : csv로 저장
- `df.values` : array 형태로 데이터프레임의 값들을 출력한다.



# 3. selection and drop

- [예제 코드](https://github.com/BoostcampAITech/lecture-note-python-basics-for-ai/blob/main/codes/pandas/%231/5_data_selection.ipynb)

- 1개의 컬럼을 추출 : `df.col1` or `df['col1']`

  - 이 경우 데이터의 반환 형태가 <u>**Series**</u>

- 여러개의 컬럼 추출 : `df[['col1','col2]]`

  - 이 경우 데이터의 반환 형태가 <u>**DataFrame**</u>

- selection : 

  - `df[조건1]` : 조건1의 부울이 True인 조건에 해당하는 df의 값들이 출력된다. 

- basic, loc, iloc selection

  ```python
  df[["name", "street"]][:2]
  df.loc[[211829, 320563], ["name", "street"]]
  df.iloc[:10, :3]
  ```





