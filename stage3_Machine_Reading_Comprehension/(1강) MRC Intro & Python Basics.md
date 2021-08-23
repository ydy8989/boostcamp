# (1강) MRC Intro & Python Basics

**강의소개**

P stage 3 기계독해 강의에 오신걸 환영합니다. 첫 강의에서는 기계독해에 대한 소개 기본적인 파이썬 지식들에 관한 강의입니다. 기계독해란 무엇인지, 어떠한 종류가 있는지, 평가는 어떻게 해야할 지에 대해 알아보고, 자연어 처리에 필요한 unicode / tokenization 개념을 배울 예정입니다. 마지막으로 한국어 기계독해 데이터인 KorQuAD를 살펴보며 실제 기계독해 데이터가 어떠한 형태로 이루어 졌는지 배워보겠습니다. 

 

**Further Reading**

- [문자열 type에 관련된 정리글](https://kunststube.net/encoding/)
- [KorQuAD 데이터 소개 슬라이드](https://www.slideshare.net/SeungyoungLim/korquad-introduction)
- [Naver Engineering: KorQuAD 소개 및 MRC 연구 사례 영상](https://tv.naver.com/v/5564630)



## Introduction to MRC

### Machine Reading Comprehension의 개념

- 기계 독해 : 주어진 지문(context)를 이해하고, 주어진 질의(Query / Question)의 답변을 추론하는 문제

- 아래 그림을 보면 검색창에서 질문을 하면 답변을 얻는 것과 약간 상이한데, 이렇게 context로부터 질문에 대한 답을 얻는 모델을 만드는데, 이 때 context를 방대한 자료로 가정하기 때문

	![image](https://user-images.githubusercontent.com/38639633/130466717-fecf0842-7419-43c2-9715-3f82d914dde7.png)



- 하지만 이렇게 방대한 양에서 검색하는 것은 매우 비효율적임
- 따라서 
	- 어떤 지문에서 답을 찾을 수 있는지 sub context를 찾고
	- 그 subcontext에서 답을 찾는 과정으로 진행
- 이는 **Dialog system**에서도 활용되는 방식



### MRC의 종류

**1. Extractive Answer Datasets**

: 질의에 대한 답이 항상 주어진 지문의 segment로 존재. 즉 추출하는 방식으로 두 가지의 세부 task로 정의할 수 있다. 

- 질문에 빈칸을 뚫어놓고 해당 빈칸에 들어갈 entity를 맞추는 방식

	![image](https://user-images.githubusercontent.com/38639633/130467810-6555590b-68fd-4676-9551-03fe097b77ca.png)

- 진짜 질문 형식으로 주어지는 방식

	![image](https://user-images.githubusercontent.com/38639633/130467866-287c16fe-6874-4340-9920-bf000561d990.png)

의 두 가지 방식으로 구성된다. 



**2. Descriptive / Narrative Answer Datasets**

: 답이 지문 내에서 추출한 span이 아니라, 질의를 보고 생성 된 sentence(or free-form)의 형태

![image](https://user-images.githubusercontent.com/38639633/130468191-ce8f8f6d-e69e-4680-8da8-f5160008040c.png)

- 위와 같이 질문에 문장 형식으로 답변할 수 있는 task이다. 



**3. Multiple-choice Datasets**

: 질의에 대한 답을 여러 개의 answer candidates 중 하나로 고르는 형태

![image](https://user-images.githubusercontent.com/38639633/130468804-d58d3bf4-247a-4af5-8b79-651c194a79ba.png)



