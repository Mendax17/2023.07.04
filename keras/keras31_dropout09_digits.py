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
from sklearn.datasets import load_digits # 숫자 이미지 인식 데이터셋

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

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

#2. 모델구성(함수형) # 함수형은 모델을 다 구성하고 마지막에 함수형을 정의한다.
input1=Input(shape=(64, ))
dense1=Dense(32, activation='relu')(input1) # layer 마다 input layer를 정의해줘야한다.
drop1=Dropout(0.1)(dense1)
dense2=Dense(64, activation='sigmoid')(drop1)
drop2=Dropout(0.1)(dense2)
dense3=Dense(32, activation='relu')(drop2)
drop3=Dropout(0.1)(dense3)
dense4=Dense(16, activation='relu')(drop3)
dense5=Dense(8, activation='linear')(dense4)
output1=Dense(10, activation='softmax')(dense5)
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

y_predict=model.predict(x_test) # [3.7969347e-10 4.6464029e-05 9.9995351e-01] # 칼럼은 3개로 분리된 상태지만 값들은 [0, 0, 1]이 아닌 softmax에 의해 각 항들의 합이 1인 형태를 띄고 있음
# 따라서 이를 완전한 one hot encoding 형태인 [0, 0, 1]로 만들어 준다음 최종적으로 2 로 변환하는 작업을 거쳐야함
#print(y_predict.shape) # (30, 3)

y_predict=np.argmax(y_predict, axis=1) # axis=1 일때는 행을 비교해서 그 행에서 softmax 형태의 확률을 확인하여 다시 one hot encoding 상태전으로 되돌려준다.
y_test=np.argmax(y_test, axis=1) # [3.2756470e-10 3.9219194e-05 9.9996078e-01] --> [0, 0, 1] --> [2] --> 결과적으로 [2 1 2 0 2]

acc=accuracy_score(y_test, y_predict)
print('accuarcy_score :', acc)

# acc : 0.9791