import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model # load_model은 저장된 h5 모델 데이터를 불러오는 기능이다.
from tensorflow.keras.layers import Dense, Input # 함수형 모델은 input layer를 정해줘야한다.
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from sklearn.datasets import load_boston # 보스턴 집값에 대한 데이터

#1. 데이터
dataset=load_boston()
x=dataset.data # 집값에 영향을 끼치는 요소 # (506, 13) # 이 데이터는 13개의 칼럼(열)을 가진 506개의 데이터이다.
y=dataset.target # 가격 데이터 # (506, )

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, random_state=333)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=333)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
scaler.fit(x_train) # x값은 변하지 않고 x 데이터를 활용하여 MinMaxScaler의 전처리 조건에 맞는 가중치를 생성한다는 의미
#x_train=scaler.fit_transform(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

print(x) # [0.00000000e+00 1.80000000e-01 6.78152493e-02 ... 2.87234043e-01 1.00000000e+00 8.96799117e-02]
print(type(x)) # <class 'numpy.ndarray'>
print("최소값 :", np.min(x)) # 0.0
print("최대값 :", np.max(x)) # 1.0

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

from tensorflow.keras.callbacks import EarlyStopping # Class형식으로 구성
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)
# monitor(관측 대상), mode(최소, 최대중에서 어떤걸 찾을건지 accuracy일때는 최대 사용), patience(갱신되고 몇번 동안 갱신 안되면 반환할건지)
# restore_best_weights(earlystropping 기능이 작동 했을때 까지중 설정한 최소 or 최대 값을 저장한다 default=False), verbose(EarlyStopping이 작동했을때 값들을 보여준다)
hist=model.fit(x_train, y_train, epochs=500, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[earlyStopping]) # model.fit의 어떤 반환값을 hist에 넣는다.

# 가중치 저장(모델 저장과 똑같지만 위치를 fit 밑으로)
path='c:/Users/eagle/Downloads/bitcamp/_save/'
model.save(path+'keras29_3_save_model.h5')

#4. 평가, 예측
mse, mae=model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict=model.predict(x_test)

r2=r2_score(y_test, y_predict)
print("r2 :", r2)

# R2 : 0.8382
# R2 : 0.8133 (모델 저장)