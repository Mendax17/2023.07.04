import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/ddarung/'
# 만약 파일을 불러올때 인덱스 컬럼을 지정해주지 않으면 자동으로 생성된다.
train_csv=pd.read_csv(path + 'train.csv', index_col=0) # 글자열 + 글자열 하면 합치는게 가능하기 때문에 중복되는 주소를 path에 str로 넣어서 표시한다. # index_col는 몇번째 행을 index로 보고 데이터 칼럼에서 뺄건지를 지정하는 명령어
#train_csv=pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv=pd.read_csv(path + 'test.csv', index_col=0) # [715 rows x 9 columns]
submission=pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv) # [1459 rows x 10 columns] # pandas로 가져온 데이터는 shape가 표기된다.
print(train_csv.shape) # (1459,10) 실질적으로 count는 y축에 표기하여야 하는 부분이므로 input_dim=9

print(train_csv.columns) # column(열)들을 보여준다.
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation', 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'], dtype='object')

print(train_csv.info()) # 전체 1459개의 데이터 이지만 1457 같이 2개가 누락된걸 "결측치"라고 한다.
# 이러한 결측치에 임의의 데이터를 넣어서 연산하면 오차가 커질 수 있기 때문에 결측치가 발생한 데이터 라인은 삭제하거나 중간값등을 넣는다 # 단 데이터 수가 작을때는 삭제 방식은 치명적이다.
print(test_csv.info()) # 전체 715개의 데이터 # count 없음
print(train_csv.describe()) # hour는 0~24까지이다.

x=train_csv.drop(['count'], axis=1) # drop은 pandas에서 특정 칼럼을 제거 하는 명령어이다. # count라는 칼럼을 제거
print(x) # [1459 rows x 9 columns]
y=train_csv['count']
print(y) # count 컬럼을 출력 결과로 설정한다.
print(y.shape) # (1459, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=0)
print(x_train.shape, x_test.shape) # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape) # (1021, ) (438, )

model=Sequential()
model.add(Dense(30, input_dim=9)) #x의 덩어리 갯수 #행렬에서 열의 갯수와 같다 #열의 갯수가 우선된다 #열=컬럼,피처,특성 을 의미 # ex)환율,금리,물가지수 이런 요소들이 열
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
loss=model.evaluate(x_test, y_test) # evaluate도 batch_size가 존재함 default 32 이므로 1/1 로 나온다. # metrics=['mae']를 넣었다면 loss 반환도 2가지를 보여준다.
print('loss : ', loss)

y_predict = model.predict(x_test) # 해당 부분에서는 이미 최적의 가중치가 생성되어 있다. # test_csv은 y값이 없다 submission 제출을 위해서 따라서 predict()에 넣어선 안된다.
print(y_predict)
# 결측치로 인하여 nan값이 출력된다.

"""
def RMSE(y_test, y_predict): # RMSE를 함수로 구현한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt=루트 씌운다는 의미 mean_squared_error는 mse 연산

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
# 하지만 결측치 포함된 데이터들이 존재하기 때문에 연산이 불가하다.
# 현재까지 test.csv 데이터 submission 데이터 사용 X

# 제출할 데이터
y_submit=model.predict(test_csv)

#R2(결정계수)란 정확도를 의미
r2=r2_score(y_test,y_predict)
print("R2 : ",r2)
"""