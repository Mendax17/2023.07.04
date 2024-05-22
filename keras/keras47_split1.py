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

# 시계열 데이터를 timesteps 만큼 어떻게 잘라서 y를 만들지가 중요한 요소

a=np.array(range(1, 11))
timesteps=5 # n개씩 자를것이다.

def split_x(dataset, split_num):
    list_1 = []
    for i in range(len(dataset)-split_num+1):
        list_1.append(dataset[i : (i + split_num)])
    return np.array(list_1)

list_2=split_x(a, timesteps)

x=list_2[:, :-1] # [[1 2 3 4] [2 3 4 5] [3 4 5 6] [4 5 6 7] [5 6 7 8] [6 7 8 9]]
y=list_2[:, -1] # [5 6 7 8 9 10]

print(x,y)
print(x.shape, y.shape) # (6, 4) (6, )

x=x.reshape(6, 4, 1)

#2. 모델구성
model=Sequential() # (N, 3, 1) units의 64가 1의 dim 부분에 들어간다.
model.add(LSTM(units=32, input_shape=(4, 1)))#, return_sequences=True)) # (N, 64) 3차원으로 받아야하는데 LSTM의 결과값은 2차원으로 나오기 때문에 그 다음에 LSTM layer를 놓으면 ERROR가 발생
#model.add(LSTM(units=32, activation='relu'))
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
es=EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1)
#mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k46_'+save_name)
model.fit(x, y, epochs=1000, batch_size=1, validation_split=0.2, verbose=3, callbacks=[es])#, mcp])

#4. 평가, 예측
loss=model.evaluate(x, y)
print('loss :', loss)
y_predict=np.array([7, 8, 9, 10]) # (4, ) 형태 이므로 (1, 4, 1) 형태로 바꿔줘야함
y_predict=y_predict.reshape(1, 4, 1)
result=model.predict(y_predict)
print('[7, 8, 9, 10]의 예측 결과 :', result)