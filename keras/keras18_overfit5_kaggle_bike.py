import tensorflow as tf
import numpy as np
import seaborn as sns
import os
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/bike/'
# 만약 파일을 불러올때 인덱스 컬럼을 지정해주지 않으면 자동으로 생성된다.
train_csv=pd.read_csv(path + 'train.csv') # [10886 rows x 12 columns]
#train_csv=pd.read_csv('./_data/ddarung/train.csv')
test_csv=pd.read_csv(path + 'test.csv') # [6493 rows x 9 columns]
submission=pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

train_csv.info() # datetime만 object형식이고 나머지는 int, float형식 # 따라서 datetime에서 년-월-일 시:분:초 를 데이터 전처리를 통하여 따로 따로 분리 해줘야한다.

train_csv['datetime'] = pd.to_datetime(train_csv['datetime']) # pandas에서 datetime과 같은 형태를 년, 월, 일, 시간, 분, 초로 편리하게 변환하려면 먼저 문자열을 'datetime' 타입으로 변경해야 합니다.
test_csv['datetime'] = pd.to_datetime(test_csv['datetime'])

train_csv['year'] = train_csv['datetime'].apply(lambda x: x.year)
train_csv['month'] = train_csv['datetime'].apply(lambda x: x.month)
train_csv['day'] = train_csv['datetime'].apply(lambda x: x.day)
train_csv['hour'] = train_csv['datetime'].apply(lambda x: x.hour)

test_csv['year'] = test_csv['datetime'].apply(lambda x: x.year)
test_csv['month'] = test_csv['datetime'].apply(lambda x: x.month)   
test_csv['day'] = test_csv['datetime'].apply(lambda x: x.day)
test_csv['hour'] = test_csv['datetime'].apply(lambda x: x.hour)

train_csv.info() # train_csv에 'year', 'month', 'day', 'hour' 칼럼이 추가 되었습니다.

"""
# 데이터 나누기전에 결측치를 삭제해줘야함

###### 결측치 처리 방법1: 제거 ######
print(train_csv.isnull().sum()) # .isnull()은 train_csv 내부의 null 값들의 모임을 출력하고 .sum()은 이 null 값들의 갯수들을 다 더한다.
# 각각의 칼럼에 대한 결측치 총합을 보여준다.
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0
train_csv=train_csv.dropna() # null 값을 가진 결측치들을 모두 삭제한다.
print(train_csv.isnull().sum())
print(train_csv.shape) # (1328, 10)
"""

x=train_csv.drop(['datetime','count','casual','registered'], axis=1) # 그리고 datetime object를 칼럼으로 빼줬으므로 삭제하고 중복되지 않는 casual, registered 칼럼들도 삭제한다.
test_csv=test_csv.drop(['datetime'], axis=1)

y=train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=70)

print(x_train, y_train)

#2. 모델
model=Sequential()
model.add(Dense(32, input_shape=(8, ), activation='linear'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# model.add(Dense(256, input_dim=8, activation='linear'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련

import time # 시간을 재기위한 API
model.compile(loss='mse', optimizer='adam')
start_time=time.time() # 시작했을때 시간을 저장
#model.fit(x_train, y_train, epochs=100, batch_size=1) # 현재는 (훈련 --> 훈련 --> 훈련) 방식
hist=model.fit(x_train, y_train, epochs=300, batch_size=8, validation_split=0.25) # train 데이터의 75%를 train으로 사용하고 25% validation 데이터로 사용하겠다는뜻
# 앞으로 훈련데이터는 train data 와 validation data 로 나눠서 작업할 것 이다.
# loss 는 x_train의 이상적인 데이터 y_train 데이터와 비교한 것이고 val_loss 는 훈련시킨 데이터를 기존 데이터에 없는 결과와 비교하는 것이므로 상대적으로 더 믿을만한 지표이다.
# loss와 val_loss 중에서 최악의 상황에 대비해 더 안좋은 val_loss를 기준으로 생각해야한다.
end_time=time.time() # 끝났을때 시간을 저장

plt.figure(figsize=(9, 6)) # 표의 사이즈
plt.plot(hist.history['loss'], c='red', marker='.', label='loss') # plot함수에 x,y를 넣으면 그래프로 보여줌 # 단 x가 순차적으로 증가할때는 명시하지 않더라도 상관없다. (ex. x축을 epochs로 표현할려는 경우)
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss') # c는 color의 의미이다. # maker는 선의 모양
plt.grid() # 표에 격자를 넣는다.
plt.title('boston loss') # 타이틀명을 정한다.
plt.xlabel('epochs') # x축 이름을 명시한다.
plt.ylabel('loss') # y 이름을 명시한다.
plt.legend() # 각 선의 이름(label)을 표시해준다.
#plt.legend(loc='upper left') # 이름이 명시될 위치를 설정한다. (default='upper right')
plt.show()

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test) # 해당 부분에서는 이미 최적의 가중치가 생성되어 있다.

from sklearn.metrics import mean_squared_error,r2_score # 2개를 import # mse를 구현하는 함수 # R2(결정계수)

def RMSE(y_test, y_predict): # RMSE를 함수로 구현한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt=루트 씌운다는 의미 mean_squared_error는 mse 연산

print("RMSE : ", RMSE(y_test, y_predict))

# 제출할 데이터
y_submit=model.predict(test_csv)
print(y_submit) # y_submit 에서도 nan이 나오는것으로 결측치가 존재한다는것을 알수있다. # test_csv 에도 결측치가 존재한다는 의미(단, 제출 자료이기 때문에 결측치 삭제 X)
# 제출하기 위해 y_submit 을 submission.csv 의 count 부분에 넣어주면 된다.
print(y_submit.shape) # (1459, 10)
print(submission.shape) # (715, 1)

# .to_csv()를 사용해서 submission_0105.csv를 완성하시오.
#pd.DataFrame(y_submit).to_csv(path_or_buf=path+'submission_0105.csv', index_label=['id'], header=['count']) # 이 함수는 index 를 순차적으로 새로 설정하기 때문에 원본과 다르다.
submission['count'] = y_submit # submission의 'count'라는 컬럼에 y_submit 값을 집어넣는다.
print(submission)

submission.to_csv(path + 'submission_0108.csv')

#R2(결정계수)란 정확도를 의미
r2=r2_score(y_test,y_predict)
print("R2 : ",r2)
print("RMSE : ", RMSE(y_test, y_predict))

# R2: 0.3493 random_state=70