Wrapup 대기파일

[1일차]

- 점수 및 순위 : 아직 미제출로 인한 순위 없음
- 시도했던 것들 : torch.Dataset과 DataLoader를 구현하려 시도하였음
	- 이슈 : 
		- 이미지를 로드하면서 transforms를 활용할 시 파일을 잘 읽지 못하는 issue가 발생
		- 여러 사람들의 조언과 구글링을 통해 돌아가게 만드는데까지 성공!
	- 원인 :
		- 구글 코랩에서 과제에 사용되었던 버전과의 차이가 있음을 인지하였음. 
		- is_pil_image 함수가 최신버전에는 빠져있다. 
		- PIL.Image를 통해 읽고, transform.compose의 전처리 순서도 조정해야함

[2일차]

- 점수 및 순위 : acc : 59.5% / f1 : 0.38% 순위 65위

- 시도했던 것들 : 

	- torch.Dataset과 DataLoader 구현 완료
	- ssh를 통한 vscode 환경에 적응, 
	- ipynb가 아니라 쉘 환경에 익숙해지기 위해 파이프라인 작성중.
	- 모델 탐색(efficientnet)

	

- 피어세션 : 

	- 각자의 의견 공유
	- U stage에 대한 서로의 피드백
	- 캠퍼 분의 데이터 로더에 대한 코드 리뷰 및 이슈 해결 위한 토론
		- 전날 겪어던 문제에 대한 내용.



[3일차]

- 점수 및 순위 : acc :  61.89% / f1 : 0.42% 순위 56위
- 시도했던 것들 : 
	- transforms 조정(리사이징, 로테이트, 랜덤크롭 수치 조정)
	- effi-b0~b6까지 테스트



[4일차]

- 점수 및 순위 : 52위 / 74.6%,  0.66%

- 시도한거 :

	- 파라미터ㅇ 방식 변경과 다양한 시도에도 불구하고 성능이 안올라감

	- dataset 클래스 중 레이블링 과정에 문제가 있음을 발견

		- 'mask'가 file path에 포함되어 있는지로 구현했었는데, 이는 incorrect_mask에도 포함된다는 것을 뒤늦게 인지함..

			```python
			if 'mask' in os.path.basename(img_path):
			    ...
			```

		- ...

	- sgd와 adam 왜? 안돼? 

	- focalloss arc face loss 

		- gamma...5

	- 스케줄러 적용 

		- 코사인

[5일차]

- 61등 / 75.17% / f1 0.66 / 
- 70등 f1 스코어 기준.. 74.43% / 0.68 달성
- 파이참 ssh 환경으로 이전
	- 제시한 베이스라인 코드를 보면서 흡수중
	- focal loss 미적용
	- 스케줄러 미적용인데 사용되는 것 같음. 확인 요망



[주말]

- 시도예정 :
	- optimization custom - adamP
	- 데이터 valid 나누는거 조정
		- 전체 데이터 나누고 - 데이터셋 - transform만들고 - 로더



[6일차] 

- 등수 변동 없음
- 시도 했던 것들 : 
	- augmentation center 350,200 - randomcrop 300,150, scheduler 조정
	- 학습에 어려움을 느끼도록 하는 것이 더 좋아보임
	- colorjier 추가하는게 더 효과적



[7일차]

- 시도예정 : 
	- kfold 구현
	- 클래스 3개로 나누기(9가지로)
		- 18개 -> 9개(성별 예측되는지만 보기) -> 3개x3개로 

- 만들어야하는거 
	- 검증셋 넣고 빼고 가능하게
	- transforms을 train, val 따로 적용가능하게!

- 시도한거 : 

	- 58age filtering + 60대 데이터 뻥튀기 : 
	- 피어세션에서 들은 것을 바탕으로 데이터 증강
		- ![image-20210406231905411](../../assets/img/boostcamp/image-20210406231905411.png)
		- 
	- 스텝lr / 큰 이미지 그대로 / 끽해봐야 horizonflip정도 / focal이 좋아보인다. 
	- 확실한거 : 
		- MaskBaseDataset < MaskSplitByProfileDataset[77번]
		- focal gamma5.0 vs CE (CE가 더 잘나옴)
	- 

	- 해볼거 : 
		- 
		- 



~~ssh -i ./key root@49.50.163.238 -p 2222~~

### todo

- 결국엔 전처리다. 
	- incorrect랑 60대 늘리기
	- 노 밸리데이션으로 테스트(최적의 모델 후 ..)
- Seed 43 |Loss 0.123790 | Fold 1 of 4| Epcoh 43 of 70 | id Base | nfnet f0 | Cropped | 59 age filter | CosineAnnealingWarmRestarts 2e-3 to 7e-5 | ArcFaceLoss FocalLoss gamma=5.0
- Seed 43 |Loss 1.243411 | Fold 1 of 4| Epcoh 7 of 70 | id Base | tf_efficientnet_b3_ns | Cropped | 59 age filter | CosineAnnealingWarmRestarts 2e-3 to 7e-5 | ArcFaceLoss FocalLoss gamma=5.0 | gridshuffle
- 
- age filtering
- 

---

test records

- [ex16] 300, 300 / Adam / f1loss / CosineAnnealingWarmRestarts / effi-b3 / 20epoch / randomcrop(300,300) / val0.2
- [ex17] 300, 300 / **AdamW** / f1loss / CosineAnnealingWarmRestarts / effi-b3 / 20epoch / randomcrop(300,300) / **val0.0**
- [ex21] 300, 300 / AdamW / f1loss / CosineAnnealingWarmRestarts / effi-b3 / **53epoch** / randomcrop(300,300) / val0.2
- [ex23] 300, 300 / AdamW / **focalloss** / ~~CosineAnnealingWarmRestarts~~ / **nfnet** / 25epoch / randomcrop(300,300) / val0.2 - 개망
- [ex24] 300, 300 / AdamW / focalloss / CosineAnnealingWarmRestarts / nfnet / 25epoch / randomcrop(300,300) / val0.2
- [ex27] 300, 300 / AdamW / **f1** / CosineAnnealingWarmRestarts / nfnet / 25epoch / randomcrop(300,300) / **val0**
- [ex43] seed 777 / adamw / crossEntropy / effinet b0 / CosineAnnealingWarmRestarts  / 20 에폭 / randomcrop + resize만 /  val0.2
- 







