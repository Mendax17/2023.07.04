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
from sklearn.datasets import load_digits # 숫자 이미지 인식 데이터셋

save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

#1. 데이터
dataset=load_digits()
x=dataset.data # (1797, 64) # 4가지 칼럼(input_dim=4)으로 결과 y(output_dim=1)는 Iris-Setosa, Iris-Versicolour, Iris-Virginica (0, 1, 2) 3가지로 구분한다.
y=dataset.target # (1797, ) # [0 ~ 0 1 ~ 1 2 ~ 2 3 ~ 3 4 ~ 4 5 ~ 5 6 ~ 6 7 ~ 7 8 ~ 8 9 ~ 9]
# x=dataset['data']
# y=dataset['target']
print(dataset.feature_names)
print(np.unique(y, return_counts=True)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 내부에 어떤 종류의 값들이 있는지 집합으로 확인 가능하다. 분류임을 확인할 수 있음
# return_counts를 True로 했을때 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180])

# plt.gray()
# plt.matshow(dataset.images[5])
# plt.show()

# One_hot_Encoding 방법1
from tensorflow.keras.utils import to_categorical # tensorflow에서 제공하는 데이터를 one hot encoding 해주는 기능이다.
y=to_categorical(y) # (1797, 10)

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, stratify=y,random_state=333)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=333)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
scaler.fit(x_train) # x값은 변하지 않고 x 데이터를 활용하여 MinMaxScaler의 전처리 조건에 맞는 가중치를 생성한다는 의미
#x_train=scaler.fit_transform(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

print(x_train.shape, x_test.shape, x_val.shape) # (1257, 64) (432, 64) (108, 64) -> (1257, 64, 1, 1)

x_train=x_train.reshape(-1, 64, 1)
x_test=x_test.reshape(-1, 64, 1)
x_val=x_val.reshape(-1, 64, 1)

#2. 모델구성
model=Sequential()
model.add(Conv1D(32, kernel_size=2, input_shape=(64, 1), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv1D(64, kernel_size=2, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Conv1D(32, kernel_size=2, activation='relu'))
model.add(Dropout(0.1))
model.add(Conv1D(16, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='linear'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k51_Conv1D_09_digits_'+save_name)
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('(loss :', loss, end=')')
print('(accuracy :', accuracy, ')')

y_predict=model.predict(x_test) # [3.7969347e-10 4.6464029e-05 9.9995351e-01] # 칼럼은 3개로 분리된 상태지만 값들은 [0, 0, 1]이 아닌 softmax에 의해 각 항들의 합이 1인 형태를 띄고 있음
# 따라서 이를 완전한 one hot encoding 형태인 [0, 0, 1]로 만들어 준다음 최종적으로 2 로 변환하는 작업을 거쳐야함
#print(y_predict.shape) # (30, 3)

y_predict=np.argmax(y_predict, axis=1) # axis=1 일때는 행을 비교해서 그 행에서 softmax 형태의 확률을 확인하여 다시 one hot encoding 상태전으로 되돌려준다.
y_test=np.argmax(y_test, axis=1) # [3.2756470e-10 3.9219194e-05 9.9996078e-01] --> [0, 0, 1] --> [2] --> 결과적으로 [2 1 2 0 2]

acc=accuracy_score(y_test, y_predict)
print('accuarcy_score :', acc)

# accuarcy_score : 0.9768518518518519