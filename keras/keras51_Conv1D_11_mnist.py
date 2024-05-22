import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Conv2D, Flatten, SimpleRNN, LSTM, GRU, MaxPooling2D, AveragePooling2D, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from tensorflow.keras.datasets import mnist # tensorflow 숫자 사진 데이터셋

save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

#1. 데이터
(x_train, y_train), (x_test, y_test)=mnist.load_data() # 이미 train과 test가 분리되어 있는 데이터셋
print(x_train.shape, y_train.shape) # (60000, 28 , 28) # x는 60000만장 가로 28 세로 28 생략된 색깔 1 이렇게 생각할 수 있다. 따라서 reshape로 (28, 28, 1)로 바꿔준다.
print(x_test.shape, y_test.shape) # (10000, 28, 28)

# reshape로 현재 3차원 데이터를 4차원 데이터로 바꿔준다.
x_train=x_train.reshape(60000, 28*28, 1)
x_test=x_test.reshape(10000, 28*28, 1)

print(x_train.shape, y_train.shape) # (60000, 28 , 28, 1) (60000, ) # one hot을 해줬다면 (60000, 10)
print(x_test.shape, y_test.shape) # (10000, 28 , 28, 1) (10000, )

print(np.unique(y_train, return_counts=True)) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]) # 10개의 칼럼이 있다는걸 알 수 있다.

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

#2. 모델
model=Sequential()
model.add(Conv1D(filters=128, kernel_size=2, input_shape=(28*28, 1), activation='relu')) # (27, 27, 128) # kernel_size=(2, 2)로 자를시 (27, 27) 사이즈의 이미지가 128개 생성된다는 의미
model.add(Conv1D(filters=64, kernel_size=2, activation='relu')) # (26, 26, 64)
model.add(Conv1D(filters=64, kernel_size=2, activation='relu')) # (25, 25, 64)
model.add(Flatten()) # 25*25*64=40000
model.add(Dense(32, activation='relu')) # input_shape=(batch_size=60000, input_dim=40000) 실질적으로 행무시 따라서 (40000, )
                                                    # (batch_size, input_dim)
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
# categorical_crossentropy 을 줘야하지만 현재 원핫을 안해줬기 때문에 
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)#, restore_best_weights=False)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k51_Conv1D_11_mnist'+save_name)#filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val), verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('loss :', loss)
print('accuracy :', accuracy)

# loss : 0.18803466856479645
# accuracy : 0.9456250071525574