import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#한줄 라인 지우기 SHIFT + DELETE

#1. 데이터 준비
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,5,4,6])

#2. 모델구성
model=Sequential() #model을 Sequential 모델로 설정한다.
model.add(Dense(3,input_dim=1))# input_dim의 1을 1덩어리로 생각 x라는 배열 한덩어리가 input data로 들어간다는 의미 x라는 배열속 숫자 하나씩 들어가는게 아니라 x가 통째로 들어감
#단 이 경우 x라는 input array의 수가 매우 커질경우 노드와 가중치를 곱하는 과정에서 오버플로가 발생할 수 있음 (성능도 좋지는 않음)
# 따라서 batch_size로 끊어서 훈련시킴
model.add(Dense(3))#hidden layer
model.add(Dense(4))#hidden layer
model.add(Dense(4))#hidden layer
model.add(Dense(4))#hidden layer
model.add(Dense(4))#hidden layer
model.add(Dense(4))#hidden layer
model.add(Dense(2))#hidden layer
model.add(Dense(1))#output layer
#node의 갯수를 늘리거나 hidden layer를 추가하는것으로 input, output, epochs를 건드리지않고 정확도를 높을수있다.

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=10, batch_size=2) # batch_size는 input array를 몇개씩 끊어서 훈련시킬지 결정 기본 default value는 32

#4. 평가, 예측
results = model.predict([6])
print('6의 예측값 : ',results)

"""
batch_size=1 일때는 배열중 1개씩 끊어서 훈련을 시키므로 이 경우 6번
batch_size=2 일때는 배열중 2개씩 끊어서 훈련을 시키므로 이 경우 3번
batch_size=3 일때는 배열중 2개씩 끊어서 훈련을 시키므로 이 경우 2번
batch_size=4 일때는 배열중 2개씩 끊어서 훈련을 시키므로 이 경우 2번
batch_size=5 일때는 배열중 2개씩 끊어서 훈련을 시키므로 이 경우 2번
batch_size=6 일때는 배열중 2개씩 끊어서 훈련을 시키므로 이 경우 1번
batch_size=7 일때는 배열중 2개씩 끊어서 훈련을 시키므로 이 경우 1번
"""