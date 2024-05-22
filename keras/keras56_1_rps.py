import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
import os # Operating System의 약자로서 운영체제에서 제공되는 여러 기능을 파이썬에서 수행할 수 있게 해줍니다. 파일이나 디렉토리 조작이 가능하고, 파일의 목록이나 path를 얻을 수 있거나, 새로운 파일 혹은 디렉토리를 작성하는 것도 가능합니다.
import glob # glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다.
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Conv2D, Flatten, SimpleRNN, LSTM, GRU, MaxPool1D, MaxPool2D, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Bidirectional, concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # ReduceLROnPlateau 는 n번 동안 갱신이 없으면 멈추는게 아니라 러닝 레이트를 설정한 비율만큼 줄여주는 기능
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 데이터 전처리
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

data_path='c:/Users/eagle/Downloads/bitcamp/AI/_data/rps/'
save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

#1. 데이터
datagen=ImageDataGenerator(rescale=1./255)

x_data=datagen.flow_from_directory(data_path, target_size=(200, 200), batch_size=2520, class_mode='categorical', color_mode='rgb', shuffle=True) # 폴더로 부터 데이터를 받아오는 함수

# print(x_data[0][0].shape) # (2520, 200, 200, 3)
# print(x_data[0][1].shape) # (2520, )
# print(x_data[0][1]) # [1. 0. 0. ... 2. 0. 1.]

x_train, x_input, y_train, y_input= train_test_split(x_data[0][0], x_data[0][1], train_size=0.8, shuffle=True, random_state=333)
x_test, x_val, y_test, y_val= train_test_split(x_input, y_input, train_size=0.8, shuffle=True, random_state=333)
# print(x_test.shape, x_val.shape) # (403, 200, 200, 3) (101, 200, 200, 3)
# print(y_test.shape, y_val.shape) # (403, 3) (101, 3)

#2. 모델
model=Sequential()
model.add(Conv2D(64, (2,2), input_shape=(200, 200, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax')) # y값이 ad폴더의 이미지인지 normal폴더의 이미지인지에 대한 데이터를 가지고 있기 때문에 0, 1만 가지고 있다. # 따라서 결과값을 sigmoid로 0 또는 1로 한정시켜준다.

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# hist=model.fit_generator(xy_train, steps_per_epoch=16, epochs=10, validation_data=xy_test, validation_steps=4) # steps_per_epoch epoch당 배치가 몇번도는지 # batch_size는 flow_from_directory에서 설정해놓은 값으로 반영된다.
hist=model.fit(x_train, y_train, epochs=1, batch_size=4, validation_data=(x_val, y_val))
# batch_size=1000으로 xy_train을 통배치 형태로 바꾸어줬으므로 xy_train[0][0]은 모든 x값을 가지고 있다.

loss=hist.history['loss'] # fit에서 훈련시킨 모든 epoch당 loss를 list 형태로 가지고 있다.
accuracy=hist.history['acc']
val_acc=hist.history['val_acc']
val_loss=hist.history['val_loss']

y_predict=model.predict(x_test)
print('result :', y_predict)

print('loss :', loss[-1])
print('accuracy :', accuracy[-1])
print('val_acc :', val_acc[-1])
print('val_loss :', val_loss[-1])

# loss : 1.361496090888977
# accuracy : 0.3239087164402008
# val_acc : 0.2871287167072296
# val_loss : 1.0993667840957642