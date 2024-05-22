#https://keras.io/api/models/model_training_apis/
import tensorflow as tf
print(tf.__version__)
import numpy as np

#1. 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1,input_dim=1)) #모델에 레이어를 쌓는다.

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=300)

#4. 평가, 예측
result = model.predict([4])
print('결과 : ',result)