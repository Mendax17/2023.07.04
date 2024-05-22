import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
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
y=train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=60)

#2. 모델
model=Sequential()
model.add(Dense(64, input_dim=9, activation='linear'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, epochs=100, batch_size=1) # 현재는 (훈련 --> 훈련 --> 훈련) 방식
hist=model.fit(x_train, y_train, epochs=300, batch_size=8, validation_split=0.25) # train 데이터의 75%를 train으로 사용하고 25% validation 데이터로 사용하겠다는뜻

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

submission.to_csv(path + 'submission_0106.csv')

#R2(결정계수)란 정확도를 의미
r2=r2_score(y_test,y_predict)
print("R2 : ",r2)
print("RMSE : ", rmse)

# RMSE : 48.20