# pipeline

mesh 데이터를 생성 및 시각화 까지 한번에 실행하는 코드

### install
root  
├── pipeline.py  
├── MotionBERT  
├── AlphaPose  
├── utils  

다음과 같이 경로를 구성함

### run

```
python pipeline.py --video D:\pose_prediction\video\boxing1.mp4
```
video만 입력을 해주면 같은 경로의 results안에 video_name의 경로에 결과가 저장이 됨

### todo

1. 기준점  
현재 video를 만들때 축을 내가 원하는 대로 설정을 하기가 어려움  
2. 여러 사람이 있을 때  
MotionBERT를 이용해서 예측을 하면 가끔 순간이동하는 것처럼 사람이 급격하게 변하는 부분이 존재함   
이는 2명의 사람이 영상 안에 잡혀있을 때 이러한 문제점이 나타남  

### video split
유투브에서 받은 동영상은 용량이 큼  
pipeline으로 실행을 하니 메모리 때문에 프로그램이 멈춤  
그래서 동영상을 분할해서 pipeline으로 mesh와 vector를 뽑아냄 