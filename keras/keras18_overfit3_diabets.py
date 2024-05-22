import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes # 당뇨병 환자 데이터

#1. 데이터
dataset=load_diabetes()
x=dataset.data #(442, 10)
y=dataset.target #(442, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=110)

#2. 모델
model=Sequential()
model.add(Dense(32, input_dim=10))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, epochs=100, batch_size=1) # 현재는 (훈련 --> 훈련 --> 훈련) 방식
hist=model.fit(x_train, y_train, epochs=400, batch_size=4, validation_split=0.20) # train 데이터의 75%를 train으로 사용하고 25% validation 데이터로 사용하겠다는뜻
# 앞으로 훈련데이터는 train data 와 validation data 로 나눠서 작업할 것 이다.
# loss 는 x_train의 이상적인 데이터 y_train 데이터와 비교한 것이고 val_loss 는 훈련시킨 데이터를 기존 데이터에 없는 결과와 비교하는 것이므로 상대적으로 더 믿을만한 지표이다.
# loss와 val_loss 중에서 최악의 상황에 대비해 더 안좋은 val_loss를 기준으로 생각해야한다.

plt.figure(figsize=(9, 6)) # 표의 사이즈
plt.plot(hist.history['loss'], c='red', marker='.', label='loss') # plot함수에 x,y를 넣으면 그래프로 보여줌 # 단 x가 순차적으로 증가할때는 명시하지 않더라도 상관없다. (ex. x축을 epochs로 표현할려는 경우)
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss') # c는 color의 의미이다. # maker는 선의 모양
plt.grid() # 표에 격자를 넣는다.
plt.title('boston loss') # 타이틀명을 정한다.
plt.xlabel('epochs') # x축 이름을 명시한다.
plt.ylabel('loss') # y 이름을 명시한다.
plt.legend() # 각 선의 이름(label)을 표시해준다.
#plt.legend(loc='upper left') # 이름이 명시될 위치를 설정한다. (default='upper right')
plt.show()

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

# R2 : 0.6105