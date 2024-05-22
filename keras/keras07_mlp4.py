import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)]) #(1,10) # (10,) (10,1) 은 똑같이 input_dim=1 로 먹힌다.
y = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],[9,8,7,6,5,4,3,2,1,0]]) #(3,10)
x=x.T #(10,1)
y=y.T #(10,3)
# input이 1개 output이 3개 연산자체는 돌아가지만 ex)비트코인 가격으로 -> 환율, 금리, 코스피 가격 예측이 불가한것과 비슷한 원리

model=Sequential()
model.add(Dense(15, input_dim=1)) #x의 덩어리 갯수 #행렬에서 열의 갯수와 같다 #열의 갯수가 우선된다 #열=컬럼,피처,특성 을 의미 # ex)환율,금리,물가지수 이런 요소들이 열
# 컬럼(관계형 데이터베이스 테이블에서 특정한 단순 자료형의 일련의 데이터값과 테이블에서의 각 열) # feature(특징)
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(14))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=1)

#4. 평가, 예측
#loss=model.evaluate(x,y) # evaluate도 batch_size가 존재함 default 32 이므로 1/1 로 나온다.
#print('loss : ', loss)
results = model.predict([[9]]) #대괄호 2개
print('예측값 : ',results) #[[10],[1.4],[0]]