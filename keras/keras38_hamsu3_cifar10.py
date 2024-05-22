import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization # Conv2D 2차원 이미지 cnn 연산 1차원은 Conv1D # Flatten 차원을 내려서 펴준다.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from tensorflow.keras.datasets import cifar10 # 흑백

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

#1. 데이터
(x_train, y_train), (x_test, y_test)=cifar10.load_data() # 이미 train과 test가 분리되어 있는 데이터셋
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) reshape 필요 X # (50000, 1)
print(x_test.shape, y_test.shape) # (10000, 32, 32, 3) # (10000, 1)

x_train=x_train.reshape(50000, 32*32*3)
x_test=x_test.reshape(10000, 32*32*3)

print(np.unique(y_train, return_counts=True)) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])

# x_train=x_train/255. # 결과는 전부 0 과 1 사이 값으로 나올 경우는 어차피 이미지 한 픽셀은 0~255 사이 값이므로 나눠줘서 연산 가능하다.
# x_test=x_test/255.

scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=x_train.reshape(50000, 32, 32, 3)
x_test=x_test.reshape(10000, 32, 32, 3)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

#2. 모델구성(함수형) # 함수형은 모델을 다 구성하고 마지막에 함수형을 정의한다.
input1=Input(shape=(32, 32, 3))
dense1=Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_uniform')(input1)
batch1=BatchNormalization()(dense1)
dense2=Conv2D(filters=32, kernel_size=3, activation='relu', kernel_initializer='he_uniform')(batch1)
batch2=BatchNormalization()(dense2)
maxpool1=MaxPooling2D(pool_size=2)(batch2)
drop1=Dropout(0.25)(maxpool1)
dense3=Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_uniform')(drop1)
batch3=BatchNormalization()(dense3)
dense4=Conv2D(filters=64, kernel_size=3, activation='relu', kernel_initializer='he_uniform')(batch3)
batch4=BatchNormalization()(dense4)
maxpool2=MaxPooling2D(pool_size=2)(batch4)
drop2=Dropout(0.25)(maxpool2)
flat1=Flatten()(drop2)
dense5=Dense(512, activation='relu')(flat1)
drop3=Dropout(0.5)(dense5)
output1=Dense(10, activation='softmax')(drop3)
model = Model(inputs=input1, outputs=output1) # 시작과 끝모델을 직접 지정해준다.

#3. 컴파일, 훈련
# categorical_crossentropy 을 줘야하지만 현재 원핫을 안해줬기 때문에
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)#, restore_best_weights=False)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k38_cifar10_'+save_name)#filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('loss :', loss)
print('accuracy :', accuracy)

# loss : 0.5503449440002441
# accuracy : 0.8240000009536743