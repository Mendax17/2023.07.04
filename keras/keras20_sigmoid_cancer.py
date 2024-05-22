import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer # 유방암 데이터셋

#1. 데이터
dataset=load_breast_cancer() # dataset은 딕셔너리 형태로 data, target 등을 key값으로 하고 있다.
x=dataset['data'] # (569, 30)
y=dataset['target'] # (569, ) # target의 value값은 전부 0 or 1 의 형태로 이루어져 있다.

print(dataset)
print(dataset.DESCR)
print(dataset.feature_names) # 30개

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=70)

#2. 모델구성
model=Sequential()
model.add(Dense(64, activation='linear', input_shape=(30,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 이진분류는 0 또는 1로만 출력 되기 때문에 몇개가 맞았는지 accuracy로 표현가능하다.
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1) # 이진분류도 accuracy보다 보편적으로 val_loss를 지표로 하는게 성능이 더 좋다.
hist=model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[earlyStopping], verbose=1)

#4. 평가, 예측
# loss=model.evaluate(x_test, y_test)
# print('loss, accuracy : ', loss) # [0.21629159152507782, 0.8859649300575256] 전자는 binary cross entropy에 대한 loss이고 후자는 accuracy에 대한 loss이다.
loss, accuracy=model.evaluate(x_test, y_test)
print('(loss :', loss, end=')')
print('(accuracy :', accuracy, ')')

y_predict=model.predict(x_test)
#print(y_predict.shape) # (114, 1)
#y_predict=y_predict.flatten() # 차원 펴주기
#print(y_predict_flat.shape) # (114, )
y_predict=np.where(y_predict > 0.5, 1, 0) # 0.5 이상이면 1 이하일때는 0 을 출력하는 문법 # [1 1 1 1 0 1 0 0 1 1]
#y_predict=[1 if index > 0.5 else 0 for index in y_predict] # 방법2 # [1, 1, 1, 1, 0, 1, 0, 0, 1, 1]

print(y_predict[:10]) #[[9.9776959e-01], ~ ,[9.3264139e-01]
print(y_test[:10]) #[1 1 1 1 0 1 0 0 1 1]

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_predict)
print('accuarcy_score :', acc)