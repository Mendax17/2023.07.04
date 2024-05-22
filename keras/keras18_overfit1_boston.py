import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import seaborn as sns
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston # 보스턴 집값에 대한 데이터

#1. 데이터
dataset=load_boston()
x=dataset.data # 집값에 영향을 끼치는 요소 # (506, 13) # 이 데이터는 13개의 칼럼(열)을 가진 506개의 데이터이다.
y=dataset.target # 가격 데이터 # (506, )

print(x.shape, y.shape) # (506, 13) (506, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=333)

#2. 모델구성
model=Sequential()
#model.add(Dense(32, input_dim=13)) # default activation="linear"
model.add(Dense(5, input_shape=(13,))) # 앞으로 다차원이 나오게 되면 input_dim이 아닌 input_shape로 하게된다. # input_dim=13과 같은 의미
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, verbose=1) # model.fit의 어떤 반환값을 hist에 넣는다.
print("======================================================================")
print(hist) # <keras.callbacks.History object at 0x00000279CCF8FDC0>
print("======================================================================")
print(hist.history) # {'loss': [34390.90625, ~ ,38.8656463623047,] val_loss': [460.1453857421875, ~ ,47.287174224853516]} # key값이 {loss, val_loss} 이고 value값이 [] 인 딕셔너리 형태로 이루어져 있다.
# 이 데이터를 이용하여 시각화 그래프를 그릴 수 있다.
print("======================================================================")
print(hist.history['loss'])
print("======================================================================")
print(hist.history['val_loss'])
print("======================================================================")

plt.figure(figsize=(9, 6)) # 표의 사이즈
plt.rcParams['font.family'] = 'Malgun Gothic' # 한글 폰트 적용
plt.plot(hist.history['loss'], c='red', marker='.', label='loss') # plot함수에 x,y를 넣으면 그래프로 보여줌 # 단 x가 순차적으로 증가할때는 명시하지 않더라도 상관없다. (ex. x축을 epochs로 표현할려는 경우)
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss') # c는 color의 의미이다. # maker는 선의 모양
plt.grid() # 표에 격자를 넣는다.
plt.title('보스턴 손실함수') # 타이틀명을 정한다.
plt.xlabel('epochs') # x축 이름을 명시한다.
plt.ylabel('loss') # y 이름을 명시한다.
plt.legend() # 각 선의 이름(label)을 표시해준다.
#plt.legend(loc='upper left') # 이름이 명시될 위치를 설정한다. (default='upper right')
plt.show()

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)