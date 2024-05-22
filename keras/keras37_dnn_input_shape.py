import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D # Maxpooling2D 는 연산할때 kenel중 가장 높은 특성값으로만 추출하는 방식이다.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from tensorflow.keras.datasets import mnist # tensorflow 숫자 사진 데이터셋

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

#1. 데이터
(x_train, y_train), (x_test, y_test)=mnist.load_data() # 이미 train과 test가 분리되어 있는 데이터셋
print(x_train.shape, y_train.shape) # (60000, 28 , 28) # x는 60000만장 가로 28 세로 28 생략된 색깔 1 이렇게 생각할 수 있다. 따라서 reshape로 (28, 28, 1)로 바꿔준다.
print(x_test.shape, y_test.shape) # (10000, 28, 28)

# MinMaxScaler 처리
x_train=x_train/255.
x_test=x_test/255.

print(x_train.shape, y_train.shape) # (60000, 28 , 28) (60000, )
print(x_test.shape, y_test.shape) # (10000, 28 , 28) (10000, )

print(np.unique(y_train, return_counts=True)) # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]) # 10개의 칼럼이 있다는걸 알 수 있다.

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.8, random_state=333)

#2. 모델
model=Sequential()
model.add(Dense(128, input_shape=(28, 28), activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten()) # dnn은 3차원 이상의 자료도 받을 수 있다. 단 이 경우 모델단에서 결과가 나오기 전까지 1차원으로 Flatten을 사용하여 내려줘야한다.
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
#model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

#3. 컴파일, 훈련
# categorical_crossentropy 을 줘야하지만 현재 원핫을 안해줬기 때문에
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)#, restore_best_weights=False)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k36_dnn1_mnist_'+save_name)#filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_val, y_val), verbose=1, callbacks=[es, mcp])

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('loss :', loss)
print('accuracy :', accuracy)

# dnn 방식
# loss : 0.33622458577156067
# accuracy : 0.8801249861717224