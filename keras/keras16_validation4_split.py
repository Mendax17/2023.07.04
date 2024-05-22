import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
x=np.array(range(1,17))
y=np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8)

#2. 모델
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, epochs=100, batch_size=1) # 현재는 (훈련 --> 훈련 --> 훈련) 방식
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25) # train 데이터의 75%를 train으로 사용하고 25% validation 데이터로 사용하겠다는뜻
# 앞으로 훈련데이터는 train data 와 validation data 로 나눠서 작업할 것 이다.
# loss 는 x_train의 이상적인 데이터 y_train 데이터와 비교한 것이고 val_loss 는 훈련시킨 데이터를 기존 데이터에 없는 결과와 비교하는 것이므로 상대적으로 더 믿을만한 지표이다.
# loss와 val_loss 중에서 최악의 상황에 대비해 더 안좋은 val_loss를 기준으로 생각해야한다.

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)

result=model.predict([17])
print("17의 예측값 : ", result)
