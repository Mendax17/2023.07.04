import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, SimpleRNN, LSTM, GRU # LSTM은 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

#1. 데이터
x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]]) # (13, 3)
y=np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70]) # (13, )
print(x.shape, y.shape)

x=x.reshape(13, 3, 1) # (13, 3, 1)
print(x)

#2. 모델구성
model=Sequential()
#model.add(SimpleRNN(units=10, input_shape=(3, 1)))
                                        # (N, 3, 1) -> (batch(행, batch 단위로 훈련을 시킴), timesteps, feature(몇개씩 훈련을 시킬 것인지))
model.add(LSTM(units=10, input_shape=(3, 1)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.summary()

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k45_1_LSTM_scale_'+save_name)
model.fit(x, y, epochs=100, batch_size=8, validation_split=0.2, verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
loss=model.evaluate(x, y)
print('loss :', loss)
y_predict=np.array([50, 60, 70]) # (3, ) # 단, 이 형태면 input_shape의 형태와는 다르므로 ERROR 발생 이를 input_shape의 [[50][60][70]]=(None, 3, 1) 형태로 바꿔줘야한다.
y_predict=y_predict.reshape(1, 3, 1)
result=model.predict(y_predict)
print('[50, 60, 70]의 예측 결과 :', result)