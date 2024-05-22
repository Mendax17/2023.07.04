import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from sklearn.datasets import fetch_california_housing # 켈리포니아 집값 데이터

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

#1. 데이터
dataset=fetch_california_housing()
x=dataset.data # (20640, 8) -> (20640, 2, 2, 2) 또는 (20640, 8, 1, 1) 등의 형태로 3차원을 바꿔줄 수 있다.
y=dataset.target # (20640, )

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, random_state=333)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=333)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
scaler.fit(x_train) # x값은 변하지 않고 x 데이터를 활용하여 MinMaxScaler의 전처리 조건에 맞는 가중치를 생성한다는 의미
#x_train=scaler.fit_transform(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

print(x_train.shape, x_test.shape, x_val.shape) # (14447, 8) (4954, 8) (1239, 8)-> (14447, 8, 1, 1)

x_train=x_train.reshape(-1, 8, 1)
x_test=x_test.reshape(-1, 8, 1)
x_val=x_val.reshape(-1, 8, 1)

print(x) # [0.00000000e+00 1.80000000e-01 6.78152493e-02 ... 2.87234043e-01 1.00000000e+00 8.96799117e-02]
print(type(x)) # <class 'numpy.ndarray'>
print("최소값 :", np.min(x)) # 0.0
print("최대값 :", np.max(x)) # 1.0

#2. 모델구성
model=Sequential() # (N, 4, 1) units의 64가 1의 dim 부분에 들어간다.
model.add(LSTM(units=32, input_shape=(8, 1), activation='linear'))#, return_sequences=True)) # (N, 64) 3차원으로 받아야하는데 LSTM의 결과값은 2차원으로 나오기 때문에 그 다음에 LSTM layer를 놓으면 ERROR가 발생
# model.add(LSTM(units=32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=10, save_best_only=True, filepath=save_mcp_path+'k48_02_california_'+save_name)
model.fit(x_train, y_train, epochs=500, batch_size=8, validation_data=(x_val, y_val), verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
mse, mae=model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict=model.predict(x_test)

r2=r2_score(y_test, y_predict)
print("R2 :", r2)

# mse :  0.8616008758544922
# mae :  0.7341359257698059
# R2 : 0.33857045291873356