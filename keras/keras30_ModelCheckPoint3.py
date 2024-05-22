import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model # Model은 함수형 모델이다.
from tensorflow.keras.layers import Dense, Input # 함수형 모델은 input layer를 정해줘야한다.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from sklearn.datasets import load_boston # 보스턴 집값에 대한 데이터

path='c:/Users/eagle/Downloads/bitcamp/_save/'

#1. 데이터
dataset=load_boston()
x=dataset.data # 집값에 영향을 끼치는 요소 # (506, 13) # 이 데이터는 13개의 칼럼(열)을 가진 506개의 데이터이다.
y=dataset.target # 가격 데이터 # (506, )

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, random_state=0)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=0)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
scaler.fit(x_train) # x값은 변하지 않고 x 데이터를 활용하여 MinMaxScaler의 전처리 조건에 맞는 가중치를 생성한다는 의미
#x_train=scaler.fit_transform(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

#2. 모델구성(함수형) # 함수형은 모델을 다 구성하고 마지막에 함수형을 정의한다.
input1=Input(shape=(13, ))
dense1=Dense(32, activation='relu')(input1) # layer 마다 input layer를 정의해줘야한다.
dense2=Dense(64, activation='sigmoid')(dense1)
dense3=Dense(32, activation='relu')(dense2)
dense4=Dense(16, activation='linear')(dense3)
dense5=Dense(8, activation='linear')(dense4)
output1=Dense(1, activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1) # 시작과 끝모델을 직접 지정해준다.
model.summary() # 5313 으로 두모델의 연산량이 같은걸 알 수 있다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)#, restore_best_weights=False)
# restore_best_weights=False 일 경우는 patience=20 만큼 미끄러진 지점에서 저장하기 때문에 더 안좋다. # 다만 restore_best_weights=False 경우일때가 더 좋게 나오는건 EarlyStopping은 train_data 기준으로 잡는것이고 evaluate는 test_data로 평가하기 때문에 미끄러진 부분이 더 좋다고 판단할 수도 있다.
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
# ModelCheckpoin의 verbose=1 일때 # Epoch 00025: val_loss improved from 10.33085 to 9.83417, saving model to c:/Users/eagle/Downloads/bitcamp/_save/MCP\keras30_ModelCheckPoint1.hdf5 # 어떤 epoch에서 val_loss가 개선된 값이 나와 저장했는지에 대한 정보를 알 수 있다.
hist=model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[es, mcp]) # callbacks=[es, mcp] 2개 이상은 [] list로 표현

# 모델 저장
model.save(path+'keras30_ModelCheckPoint3_save_model.h5')

#model=load_model(path+'MCP/keras30_ModelCheckPoint3.hdf5')

#4. 평가, 예측
print("======================= 1. 기본 출력 =======================")
mse, mae=model.evaluate(x_test, y_test)
print('mse : ', mse)
#print('mae : ', mae)

y_predict=model.predict(x_test)

r2=r2_score(y_test, y_predict)
print("R2 :", r2)

print("====================== 2. load_model 출력 ======================")
model2=load_model(path+'keras30_ModelCheckPoint3_save_model.h5')
mse, mae=model2.evaluate(x_test, y_test)
print('mse : ', mse)
#print('mae : ', mae)

y_predict=model2.predict(x_test)

r2=r2_score(y_test, y_predict)
print("R2 :", r2)

print("====================== 3. ModelCheckPoint 출력 ======================")
model3=load_model(path+'MCP/keras30_ModelCheckPoint3.hdf5') # 가장 좋은 checkpoint값에는 모델+가중치가 저장되어 있다는걸 알 수 있다.
mse, mae=model3.evaluate(x_test, y_test)
print('mse : ', mse)
#print('mae : ', mae)

y_predict=model3.predict(x_test)

r2=r2_score(y_test, y_predict)
print("R2 :", r2)

# MCP 저장된 값 R2 : 0.8212