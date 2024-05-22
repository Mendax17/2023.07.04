import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],[9,8,7,6,5,4,3,2,1,0]]) #(3,10)
y = np.array([2,4,6,8,10,12,14,16,18,20]) #(10,1)
print(x.shape)
print(y.shape)

x=x.T #행,열을 전치한다는 의미
print(x.shape) #(10,3)

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=3)) #x의 덩어리 갯수 #행렬에서 열의 갯수와 같다 #열의 갯수가 우선된다 #열=컬럼,피처,특성 을 의미 # ex)환율,금리,물가지수 이런 요소들이 열
# 컬럼(관계형 데이터베이스 테이블에서 특정한 단순 자료형의 일련의 데이터값과 테이블에서의 각 열) # feature(특징)
model.add(Dense(7))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x,y) # evaluate도 batch_size가 존재함 default 32 이므로 1/1 로 나온다.
print('loss : ', loss)
results = model.predict([[10,1.4,0]]) #대괄호 2개
print('[10, 1.4,0]의 예측값 : ',results)