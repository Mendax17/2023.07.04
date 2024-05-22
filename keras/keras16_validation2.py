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

x_train, x_input, y_train, y_input = train_test_split(x,y,train_size=0.7)
x_test, x_validation, y_test, y_validation = train_test_split(x_input,y_input,train_size=0.5)

# x_train=np.array(range(1,11))
# y_train=np.array(range(1,11))
# x_test=np.array([11,12,13])
# y_test=np.array([11,12,13])
# x_validation=np.array([14,15,16])
# y_validation=np.array([14,15,16])

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
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_validation, y_validation)) # (훈련 --> 평가 --> 훈련 --> 평가) 방식 단, 사이클이 늘어날수록 과적합 문제가 발생할수 있다.

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)

result=model.predict([17])
print("17의 예측값 : ", result)