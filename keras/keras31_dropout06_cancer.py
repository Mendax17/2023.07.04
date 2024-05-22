import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model # Model은 함수형 모델이다.
from tensorflow.keras.layers import Dense, Input, Dropout # Dropout 히든 레이어에서 일부를 연산하지 않는 기능
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
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

#2. 모델구성(함수형) # 함수형은 모델을 다 구성하고 마지막에 함수형을 정의한다.
input1=Input(shape=(30, ))
dense1=Dense(32, activation='linear')(input1) # layer 마다 input layer를 정의해줘야한다.
drop1=Dropout(0.1)(dense1)
dense2=Dense(64, activation='relu')(drop1)
drop2=Dropout(0.1)(dense2)
dense3=Dense(32, activation='relu')(drop2)
drop3=Dropout(0.1)(dense3)
dense4=Dense(16, activation='relu')(drop3)
dense5=Dense(8, activation='relu')(dense4)
output1=Dense(1, activation='sigmoid')(dense5)
model = Model(inputs=input1, outputs=output1) # 시작과 끝모델을 직접 지정해준다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)#, restore_best_weights=False)

date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
#performance_info='({val_loss:.2f}-{epoch:04d})' # 04d 는 정수 4자리까지 표시하겠다는 의미이다. # .2f는 소수점 2번째 자리까지 표시하라는 의미이다. # 0037-0.0048.hdf5 # ModelCheckpoint에서 해당 값들을 채워준다.
performance_info='({val_loss:.2f})'
save_name=date_now+performance_info+'.hdf5'

mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k31_'+save_name)#filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
# ModelCheckpoin의 verbose=1 일때 # Epoch 00025: val_loss improved from 10.33085 to 9.83417, saving model to c:/Users/eagle/Downloads/bitcamp/_save/MCP\keras30_ModelCheckPoint1.hdf5 # 어떤 epoch에서 val_loss가 개선된 값이 나와 저장했는지에 대한 정보를 알 수 있다.
hist=model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[es, mcp]) # callbacks=[es, mcp] 2개 이상은 [] list로 표현

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

# acc : 0.9705