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
from tensorflow.keras.datasets import fashion_mnist # 10가지의 옷 종류를 분류하는 방법

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data() # 이미 train과 test가 분리되어 있는 데이터셋
print(x_train.shape, y_train.shape) # (60000, 28 ,28) (60000, )
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000, ) # 10개의 옷종류중에서 구별하는 방법

x_train=x_train.reshape(-1, 784) # (60000, 784)
x_test=x_test.reshape(-1, 784) # (8000, 784)

print(x_train[0])
print(y_train[0])

# plt.imshow(x_train[10], 'gray')
# plt.show()

print(np.unique(y_train, return_counts=True)) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000])

scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

#2. 모델
model=Sequential()
model.add(Dense(128, input_shape=(784, ), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
# categorical_crossentropy 을 줘야하지만 현재 원핫을 안해줬기 때문에
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)#, restore_best_weights=False)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k36_dnn2_fashion_'+save_name)#filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), verbose=3, callbacks=[es, mcp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('loss :', loss)
print('accuracy :', accuracy)

# dnn 방식
# loss : 0.3252016603946686
# accuracy : 0.887499988079071