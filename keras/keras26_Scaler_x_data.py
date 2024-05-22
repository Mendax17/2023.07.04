import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler # 데이터 전처리
from sklearn.datasets import load_boston # 보스턴 집값에 대한 데이터

#1. 데이터
dataset=load_boston()
x=dataset.data # 집값에 영향을 끼치는 요소 # (506, 13) # 이 데이터는 13개의 칼럼(열)을 가진 506개의 데이터이다.
y=dataset.target # 가격 데이터 # (506, )

from sklearn.preprocessing import MinMaxScaler # 데이터 전처리 # 최대값으로 나눠주겠다.
scaler = StandardScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
scaler.fit(x) # x값은 변하지 않고 x 데이터를 활용하여 MinMaxScaler의 전처리 조건에 맞는 가중치를 생성한다는 의미
x=scaler.transform(x)
print(x) # [0.00000000e+00 1.80000000e-01 6.78152493e-02 ... 2.87234043e-01 1.00000000e+00 8.96799117e-02]
print(type(x)) # <class 'numpy.ndarray'>
print("최소값 :", np.min(x)) # 0.0
print("최대값 :", np.max(x)) # 1.0

print(x.shape, y.shape) # (506, 13) (506, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=333)

#2. 모델구성
model=Sequential()
model.add(Dense(32, input_shape=(13, ), activation='linear')) # 앞으로 다차원이 나오게 되면 input_dim이 아닌 input_shape로 하게된다. # input_dim=13과 같은 의미
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping # Class형식으로 구성
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
# monitor(관측 대상), mode(최소, 최대중에서 어떤걸 찾을건지 accuracy일때는 최대 사용), patience(갱신되고 몇번 동안 갱신 안되면 반환할건지)
# restore_best_weights(earlystropping 기능이 작동 했을때 까지중 설정한 최소 or 최대 값을 저장한다 default=False), verbose(EarlyStopping이 작동했을때 값들을 보여준다)
hist=model.fit(x_train, y_train, epochs=300, batch_size=1, validation_split=0.2, verbose=1, callbacks=[earlyStopping]) # model.fit의 어떤 반환값을 hist에 넣는다.

#4. 평가, 예측
mse, mae=model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict=model.predict(x_test)

print("y_test(원래값) :", y_test)
r2=r2_score(y_test, y_predict)
print("r2 :", r2)

# r2 : 0.8382