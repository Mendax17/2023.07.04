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



x=train_csv.drop(['count'], axis=1) # drop은 pandas에서 특정 칼럼을 제거 하는 명령어이다. # count라는 칼럼을 제거
print(x) # [1459 rows x 9 columns]
y=train_csv['count']
print(y) # count 컬럼을 출력 결과로 설정한다.
print(y.shape) # (1459, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=60)
print(x_train.shape, x_test.shape) # (929, 9) (399, 9)
print(y_train.shape, y_test.shape) # (929, ) (399, )

model=Sequential()
model.add(Dense(512, input_dim=9)) #x의 덩어리 갯수 #행렬에서 열의 갯수와 같다 #열의 갯수가 우선된다 #열=컬럼,피처,특성 을 의미 # ex)환율,금리,물가지수 이런 요소들이 열
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
import time # 시간을 재기위한 API
model.compile(loss='mae', optimizer='adam')
start_time=time.time() # 시작했을때 시간을 저장
model.fit(x_train, y_train, epochs=20, batch_size=16)
end_time=time.time() # 끝났을때 시간을 저장

print("걸린시간 : ", end_time-start_time)

#4. 평가, 예측
loss=model.evaluate(x_test, y_test) # evaluate도 batch_size가 존재함 default 32 이므로 1/1 로 나온다. # metrics=['mae']를 넣었다면 loss 반환도 2가지를 보여준다.
print('loss : ', loss)

y_predict = model.predict(x_test) # 해당 부분에서는 이미 최적의 가중치가 생성되어 있다. # test_csv은 y값이 없다 submission 제출을 위해서 따라서 predict()에 넣어선 안된다.
print(y_predict)
# 결측치로 인하여 nan값이 출력된다.

def RMSE(y_test, y_predict): # RMSE를 함수로 구현한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt=루트 씌운다는 의미 mean_squared_error는 mse 연산

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)
# 하지만 결측치 포함된 데이터들이 존재하기 때문에 연산이 불가하다.
# 현재까지 test.csv 데이터 submission 데이터 사용 X

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

submission.to_csv(path + 'submission_0105.csv')

#R2(결정계수)란 정확도를 의미
r2=r2_score(y_test,y_predict)
print("R2 : ",r2)
print("RMSE : ", rmse)

# 50.46 epochs=300 batch_size=8 random_state=60
# submission_0105=pd.read_csv(path + 'submission_0105.csv', index_col=0)
# print(submission_0105.info())

# 데이콘은 평가결과를 test.csv 를 넣었을때 100프로 채점하지 않고 50프로 정도만 채점하고 private 공간에 보관하고 나중에 50% 채점하여 합산 점수를 표기하기 때문에 실제 점수보다 낮게 보인다.

# cpu 걸린시간 : 1.4
# gpu 걸린시간 : 5.5

# 현재는 dnn 모델 cnn 모델은 GPU 연산이 거의 필수