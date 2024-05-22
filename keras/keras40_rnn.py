import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, SimpleRNN # simpleRNN은 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

#1. 데이터
dataset=np.array([1,2,3,4,5,6,7,8,9,10]) # rnn은 dnn의 y값 즉 test데이터가 없다 직접 나눠서 y값을 만들어야함 # (10, )

x=np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9]]) # (7, 3) # 1자로 되어있는 시계열 데이터를 3일씩 잘랐다.
y=np.array([4, 5, 6, 7, 8, 9, 10]) # (7, )

x=x.reshape(7, 3, 1) # (7, 3, 1) # dnn은 2차원이상 cnn은 4차원 rnn은 3차원
print(x) # [[[1][2][3]] [[2][3][4]] [[3][4][5]] [[4][5][6]] [[5][6][7]] [[6][7][8]] [[7][8][9]]]
# 1->2 이고 2->3 이다. 이런식으로 연산이 진행되어야 하는데 input_shape 형태를 기존의 [1,2,3]=(3, ) 형태로 넣어준다면 [1,2,3] 통으로 X로 보고 4라는 y를 찾아내는 연산이 되기 때문에
# [[1][2][3]] 같은 (3, 1) 형태로 만들어줘야 1->2 이고 2->3 이고 따라서 3->4 이다. 라는 연산이 된다.

#2. 모델구성
model=Sequential()
model.add(SimpleRNN(64, input_shape=(3, 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss=model.evaluate(x, y)
print('loss :', loss)
y_predict=np.array([8, 9, 10]) # (3, ) # 단, 이 형태면 input_shape의 형태와는 다르므로 ERROR 발생 이를 input_shape의 [[8][9][10]]=(None, 3, 1) 형태로 바꿔줘야한다.
y_predict=y_predict.reshape(1, 3, 1)
result=model.predict(y_predict)
print('[8, 9, 10]의 예측 결과 :', result)

# loss : 0.7129192352294922
# [8, 9, 10]의 예측 결과 : [[8.572866]]