import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]) #input_dim=2 인 상태, 이 데이터 형태는 잘못된 형태, 데이터 덩어리가 2덩이라는 정도로만 이해
# [[1,1],[2,1],[3,1],[4,1],[5,2],[6,1.3],[7,1.4],[8,1.5],[9,1.6],[10,1.4]] 같은 형식이 옳은 형태
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape) #(2,10) 
print(y.shape) #(10,)

x=x.T #행,열을 전치한다는 의미
print(x.shape) #(10,2) #[[1,1],[2,1],[3,1],[4,1],[5,2],[6,1.3],[7,1.4],[8,1.5],[9,1.6],[10,1.4]] 형태

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=2)) #x의 덩어리 갯수 #행렬에서 열의 갯수와 같다 #열의 갯수가 우선된다 #열=컬럼,피처,특성 을 의미 # ex)환율,금리,물가지수 이런 요소들이 열
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x,y) #input data, output data를 넣어 평가한다. 실제 평가데이터는 훈련시킨 데이터가 들어가면 안된다.(좋은 값이 나올 확률이 높음)
print('loss : ', loss)
results = model.predict([[10,1.4]]) #대괄호 2개
print('[10, 1.4]의 예측값 : ',results)

"""
결과: [[20.065834]] 훈련시킨 y 데이터값이 20 이므로 거의 정확하다.
"""