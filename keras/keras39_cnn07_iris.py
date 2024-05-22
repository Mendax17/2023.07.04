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
from sklearn.datasets import load_iris # 꽃에 대한 4가지 칼럼 정보를 보고 꽃을 맞추는 데이터셋

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

#1. 데이터
dataset=load_iris()
x=dataset.data
y=dataset.target

# One_hot_Encoding 방법1
from tensorflow.keras.utils import to_categorical # tensorflow에서 제공하는 데이터를 one hot encoding 해주는 기능이다.
y=to_categorical(y) # (150, 3)

# One_hot_Encoding 방법2
# y=tf.one_hot(y, 3) # (150, 3) # One-hot-Encoding

print(dataset.DESCR) # pands.describe() / .info()
# ============== ==== ==== ======= ===== ====================
#                 Min  Max   Mean    SD   Class Correlation
# ============== ==== ==== ======= ===== ====================
# sepal length:   4.3  7.9   5.84   0.83    0.7826
# sepal width:    2.0  4.4   3.05   0.43   -0.4194
# petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)     # Class Correlation가 높은 데이터는 비슷한 데이터라는 뜻으로 둘중 하나가 없어도 크게 차이가 없을 수 도 없다는 의미이다.
# petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)     # 무조건 데이터가 많다고 좋은것은 아니다. 오히려 연관성이 떨어지는 데이터는 모델 성능 자체를 낮출수도 있다.
# ============== ==== ==== ======= ===== ====================
print(dataset.feature_names) # pands.columns

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, stratify=y,random_state=333)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=333)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
scaler.fit(x_train) # x값은 변하지 않고 x 데이터를 활용하여 MinMaxScaler의 전처리 조건에 맞는 가중치를 생성한다는 의미
#x_train=scaler.fit_transform(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

print(x_train.shape, x_test.shape, x_val.shape) # (105, 4) (36, 4) (9, 4) -> (105, 4, 1, 1)

x_train=x_train.reshape(-1, 4, 1, 1)
x_test=x_test.reshape(-1, 4, 1, 1)
x_val=x_val.reshape(-1, 4, 1, 1)

#2. 모델구성
model=Sequential()
model.add(Conv2D(32, kernel_size=(2, 1), input_shape=(4, 1, 1), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(64, kernel_size=(2, 1), activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=(2, 1), activation='relu'))
model.add(Dropout(0.1))
#model.add(Conv2D(16, kernel_size=(2, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k39_cnn07_iris_'+save_name)
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('(loss :', loss, end=')')
print('(accuracy :', accuracy, ')')

from sklearn.metrics import accuracy_score
y_predict=model.predict(x_test) # [3.7969347e-10 4.6464029e-05 9.9995351e-01] # 칼럼은 3개로 분리된 상태지만 값들은 [0, 0, 1]이 아닌 softmax에 의해 각 항들의 합이 1인 형태를 띄고 있음
# 따라서 이를 완전한 one hot encoding 형태인 [0, 0, 1]로 만들어 준다음 최종적으로 2 로 변환하는 작업을 거쳐야함
#print(y_predict.shape) # (30, 3)

y_predict=np.argmax(y_predict, axis=1) # axis=1 일때는 행을 비교해서 그 행에서 softmax 형태의 확률을 확인하여 다시 one hot encoding 상태전으로 되돌려준다.
y_test=np.argmax(y_test, axis=1) # [3.2756470e-10 3.9219194e-05 9.9996078e-01] --> [0, 0, 1] --> [2] --> 결과적으로 [2 1 2 0 2] # y_test는 one hot encoding을 해주지 않았을때는 argmax를 돌리면 Error 발생

# print(y_predict[:5])
# print(y_test[:5])

acc=accuracy_score(y_test, y_predict)
print('accuarcy_score :', acc)

# accuarcy_score : 0.9166666666666666