import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, SimpleRNN, LSTM, GRU, MaxPooling2D, AveragePooling2D, BatchNormalization, Bidirectional # Bidirectional 순차 데이터를 양방향으로 사용하여 2배로 훈련시킨다는 의미
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

# return_sequences를 엮어보자

a=np.array(range(1, 101)) # [1, ~ ,100]
x_predict=np.array(range(96, 106)) # (10, ) # 10개를 5개씩 잘라서 사용해야함
# 예상 목표 y = 100, 101, 102, 103, 104, 105, 106, 107

timesteps1 = 5 # x는 4개 , y는 1개
timesteps2 = 4 # x는 4개 , y는 1개

def split_x(dataset, split_num):
    list_1 = []
    for i in range(len(dataset)-split_num+1):
        list_1.append(dataset[i : (i + split_num)])
    return np.array(list_1)

list_2=split_x(a, timesteps1)

x=list_2[:, :-1] # [[1 2 3 4] [2 3 4 5] [3 4 5 6] [4 5 6 7] [5 6 7 8] [6 7 8 9]]
y=list_2[:, -1] # [5 6 7 8 9 10]

x_predict=split_x(x_predict, timesteps2) # (7, 4)

print(x,y)
print(x.shape, y.shape, x_predict.shape) # (96, 4) (96, ) (10, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=40) # train_test_split는 2차원만 받아들일 수 있다. # train_size의 default는 0.75

x_train=x_train.reshape(-1, 4, 1) # (72, 4, 1)
x_test=x_test.reshape(-1, 4, 1) # (24, 4, 1)
x_predict=x_predict.reshape(-1, 4, 1) # (7, 4, 1)

print(x_train.shape, x_test.shape, x_predict.shape)

#2. 모델구성
model=Sequential() # (N, 4, 1) units의 64가 1의 dim 부분에 들어간다.
model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(4, 1))) # Bidirectional에서의 return_sequences는 모델의 내부에서 정의해줘야한다.
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)
#mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k46_'+save_name)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=3, callbacks=[es])#, mcp])

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss :', loss)
result=model.predict(x_predict)
print('[100 ~ 106]의 예측 결과 :', result)