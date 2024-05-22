import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Conv2D, Flatten, SimpleRNN, LSTM, GRU, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Bidirectional, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

x1_datasets=np.array([range(100), range(301, 401)]).transpose() # transpose로 행과 열을 바꿔줬음
print(x1_datasets) # 삼성전자 시가, 고가
# [  0 301]
# [  1 302]
# [  2 303]
#     ~
# [ 97 398]
# [ 98 399]
# [ 99 400]
print(x1_datasets.shape) # (100, 2) # input_shape=(2, )

x2_datasets=np.array([range(101, 201), range(411, 511), range(150, 250)]).transpose()

print(x2_datasets.shape) # (100, 3) # 아모레 시가, 고가, 종가 # input_shape=(3, )

y=np.array(range(2001, 2101)) # (100, ) # 삼성전자의 하루뒤 종가

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_datasets, x2_datasets, y, train_size=0.7,random_state=333)

print(x1_train.shape, x1_test.shape) # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape) # (70, 3) (30, 3)
print(y_train.shape, y_test.shape) # (70,) (30,)

#2 모델구성(함수형)

#2-1. 모델1
input1=Input(shape=(2, ))
dense1=Dense(11, activation='relu', name='ds11')(input1)
dense2=Dense(12, activation='relu', name='ds12')(dense1)
dense3=Dense(13, activation='relu', name='ds13')(dense2)
output1=Dense(14, activation='relu', name='ds14')(dense3)

#2-2. 모델2
input2=Input(shape=(3, ))
dense21=Dense(21, activation='linear', name='ds21')(input2)
dense22=Dense(22, activation='linear', name='ds22')(dense21)
output2=Dense(23, activation='linear', name='ds23')(dense22)

#2-3. 모델병합
from tensorflow.keras.layers import concatenate # 사슬처럼 엮는 즉 병합하는 기능
# 앙상블은 칼럼 10개 중에서 3개 끼리 잘맞고 7개 끼리가 잘 맞는다고 생각될때 3개 끼리 연산시키고 7개 끼리 연산시킨뒤 두 모델을 합칠때 사용한다.
merge1=concatenate([output1, output2], name='mg1') # 병합모델의 input은 모델1 과 모델2 의 제일 마지막 레이어가 input으로 들어온다.
merge2=Dense(12, activation='relu', name='mg2')(merge1)
merge3=Dense(13, name='mg3')(merge2)
last_output=Dense(1, name='last')(merge3)
model = Model(inputs=[input1, input2], outputs=last_output) # 시작과 끝모델을 직접 지정해준다.

model.summary()

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)#, restore_best_weights=False)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k52_ensemble1_'+save_name)#filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
loss=model.evaluate([x1_test, x2_test], y_test)
print('loss :', loss)