import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston # 보스턴 집값에 대한 데이터

#1. 데이터
dataset=load_boston()
x=dataset.data # 집값에 영향을 끼치는 요소 # (506, 13) # 이 데이터는 13개의 칼럼(열)을 가진 506개의 데이터이다.
y=dataset.target # 가격 데이터 # (506, )

print(x.shape, y.shape) # (506, 13) (506, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=333)

#2. 모델구성
model=Sequential()
#model.add(Dense(32, input_dim=13)) # default activation="linear"
model.add(Dense(5, input_shape=(13,))) # 앞으로 다차원이 나오게 되면 input_dim이 아닌 input_shape로 하게된다. # input_dim=13과 같은 의미
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
start_time=time.time() # 시작했을때 시간을 저장
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, verbose=2) # verbose는 실제 훈련 과정을 시각적으로 보여줄지 안 보여줄지를 선택하는 인수이다.
# verbose=0(훈련 과정을 보여주지 않는다), 1(훈련과정 전부를 보여준다), 2(프로그래스바를 생략하고 보여준다), 3(epoch의 숫자만 보여준다) # 훈련시간: 0 < 3 < 2 < 1
end_time=time.time() # 끝났을때 시간을 저장

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)
print("걸린시간 : ", end_time-start_time)