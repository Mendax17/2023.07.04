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

#2. 모델구성(함수형) # 함수형은 모델을 다 구성하고 마지막에 함수형을 정의한다.
input1=Input(shape=(4, ))
dense1=Dense(32, activation='relu')(input1) # layer 마다 input layer를 정의해줘야한다.
drop1=Dropout(0.1)(dense1)
dense2=Dense(64, activation='sigmoid')(drop1)
drop2=Dropout(0.1)(dense2)
dense3=Dense(32, activation='relu')(drop2)
drop3=Dropout(0.1)(dense3)
dense4=Dense(16, activation='relu')(drop3)
dense5=Dense(8, activation='linear')(dense4)
output1=Dense(3, activation='softmax')(dense5)
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

# acc : 0.9444