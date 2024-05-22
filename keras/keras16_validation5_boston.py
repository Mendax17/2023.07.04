import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston # 보스턴 집값에 대한 데이터

#1. 데이터
dataset=load_boston()
x=dataset.data # 집값에 영향을 끼치는 요소 # (506, 13) # 이 데이터는 13개의 칼럼(열)을 가진 506개의 데이터이다.
y=dataset.target # 가격 데이터 # (506, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8)

#2. 모델
model=Sequential()
model.add(Dense(32, input_dim=13, activation='linear'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, epochs=100, batch_size=1) # 현재는 (훈련 --> 훈련 --> 훈련) 방식
model.fit(x_train, y_train, epochs=300, batch_size=8, validation_split=0.25) # train 데이터의 75%를 train으로 사용하고 25% validation 데이터로 사용하겠다는뜻
# 앞으로 훈련데이터는 train data 와 validation data 로 나눠서 작업할 것 이다.
# loss 는 x_train의 이상적인 데이터 y_train 데이터와 비교한 것이고 val_loss 는 훈련시킨 데이터를 기존 데이터에 없는 결과와 비교하는 것이므로 상대적으로 더 믿을만한 지표이다.
# loss와 val_loss 중에서 최악의 상황에 대비해 더 안좋은 val_loss를 기준으로 생각해야한다.

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) # 해당 부분에서는 이미 최적의 가중치가 생성되어 있다.

from sklearn.metrics import mean_squared_error,r2_score # 2개를 import # mse를 구현하는 함수 # R2(결정계수)

def RMSE(y_test, y_predict): # RMSE를 함수로 구현한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt=루트 씌운다는 의미 mean_squared_error는 mse 연산

print("RMSE : ", RMSE(y_test, y_predict))

#R2(결정계수)란 정확도를 의미
r2=r2_score(y_test,y_predict)
print("R2 : ",r2)

# R2 : 0.77