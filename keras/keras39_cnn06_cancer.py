import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten # Conv2D 2차원 이미지 cnn 연산 1차원은 Conv1D # Flatten 차원을 내려서 펴준다.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from sklearn.datasets import load_breast_cancer # 유방암 데이터셋

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

#1. 데이터
dataset=load_breast_cancer() # dataset은 딕셔너리 형태로 data, target 등을 key값으로 하고 있다.
x=dataset.data
y=dataset.target

print(dataset)
print(dataset.DESCR)
print(dataset.feature_names) # 30개

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, random_state=70)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=70)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
scaler.fit(x_train) # x값은 변하지 않고 x 데이터를 활용하여 MinMaxScaler의 전처리 조건에 맞는 가중치를 생성한다는 의미
#x_train=scaler.fit_transform(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

print(x_train.shape, x_test.shape, x_val.shape) # (398, 30) (136, 30) (35, 30) -> (398, 30, 1, 1)

x_train=x_train.reshape(-1, 30, 1, 1)
x_test=x_test.reshape(-1, 30, 1, 1)
x_val=x_val.reshape(-1, 30, 1, 1)

#2. 모델구성
model=Sequential()
model.add(Conv2D(32, kernel_size=(2, 1), input_shape=(30, 1, 1), activation='linear'))
model.add(Dropout(0.1))
model.add(Conv2D(64, kernel_size=(2, 1), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=(2, 1), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(16, kernel_size=(2, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k39_cnn06_cancer_'+save_name)
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('(loss :', loss, end=')')
print('(accuracy :', accuracy, ')')

y_predict=model.predict(x_test)
#print(y_predict.shape) # (114, 1)
#y_predict=y_predict.flatten() # 차원 펴주기
#print(y_predict_flat.shape) # (114, )
y_predict=np.where(y_predict > 0.5, 1, 0) # 0.5 이상이면 1 이하일때는 0 을 출력하는 문법 # [1 1 1 1 0 1 0 0 1 1]
#y_predict=[1 if index > 0.5 else 0 for index in y_predict] # 방법2 # [1, 1, 1, 1, 0, 1, 0, 0, 1, 1]

#print(y_predict[:10]) #[[9.9776959e-01], ~ ,[9.3264139e-01]
#print(y_test[:10]) #[1 1 1 1 0 1 0 0 1 1]

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_predict)
print('accuarcy_score :', acc)

# accuarcy_score : 0.9632352941176471