import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Conv2D, Flatten, SimpleRNN, LSTM, GRU, MaxPooling2D, AveragePooling2D, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

data_path='c:/Users/eagle/Downloads/bitcamp/AI/_data/bike/'
save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

#1. 데이터
# 만약 파일을 불러올때 인덱스 컬럼을 지정해주지 않으면 자동으로 생성된다.
train_csv=pd.read_csv(data_path + 'train.csv')#, index_col=0) # datetime을 가져오기 위해 인덱스 컬럼 지정 안해줌
#train_csv=pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv=pd.read_csv(data_path + 'test.csv')#, index_col=0) # [715 rows x 9 columns]
submission=pd.read_csv(data_path + 'sampleSubmission.csv', index_col=0)

train_csv['datetime'] = pd.to_datetime(train_csv['datetime'])
test_csv['datetime'] = pd.to_datetime(test_csv['datetime'])

train_csv['year'] = train_csv['datetime'].apply(lambda x: x.year)
train_csv['month'] = train_csv['datetime'].apply(lambda x: x.month)
train_csv['day'] = train_csv['datetime'].apply(lambda x: x.day)
train_csv['hour'] = train_csv['datetime'].apply(lambda x: x.hour)

test_csv['year'] = test_csv['datetime'].apply(lambda x: x.year)
test_csv['month'] = test_csv['datetime'].apply(lambda x: x.month)
test_csv['day'] = test_csv['datetime'].apply(lambda x: x.day)
test_csv['hour'] = test_csv['datetime'].apply(lambda x: x.hour)

# train_csv.set_index('datetime',inplace=True)
# test_csv.set_index('datetime',inplace=True)

# train_csv = train_csv.astype('float')
# test_csv = test_csv.astype('float')

# plt.figure(figsize = (20, 12))
# sns.lineplot(x = 'day', y = 'count', data = train_csv, hue = 'month')
# plt.show()

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

x=train_csv.drop(['datetime','count','casual','registered'], axis=1) # drop은 pandas에서 특정 칼럼을 제거 하는 명령어이다. # count라는 이름의 레이블을 제거 # axis=0 이면 가로줄 1이면 세로줄 제거
test_csv=test_csv.drop(['datetime'], axis=1) # test_csv 파일을 불러올때 index 처리를 안해줬으므로 datetime 열을 제거해 줘야한다.
y=train_csv['count']

# x=x.astype('float')
# y=y.astype('float')

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, random_state=300)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=300)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)
test_csv=scaler.transform(test_csv)

print(x_train.shape, x_test.shape, x_val.shape) # (7620, 12) (2612, 12) (654, 12) -> (7620, 12, 1, 1)

x_train=x_train.reshape(-1, 12, 1)
x_test=x_test.reshape(-1, 12, 1)
x_val=x_val.reshape(-1, 12, 1)
test_csv=test_csv.reshape(-1, 12, 1)

print(x) # [0.00000000e+00 1.80000000e-01 6.78152493e-02 ... 2.87234043e-01 1.00000000e+00 8.96799117e-02]
print(type(x)) # <class 'numpy.ndarray'>
print("최소값 :", np.min(x)) # 0.0
print("최대값 :", np.max(x)) # 1.0

#2. 모델구성
model=Sequential()
model.add(Conv1D(32, kernel_size=2, input_shape=(12, 1), activation='linear'))
model.add(Dropout(0.1))
model.add(Conv1D(64, kernel_size=2, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Conv1D(32, kernel_size=2, activation='relu'))
model.add(Dropout(0.1))
model.add(Conv1D(16, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k51_Conv1D_05_kaggle_bike_'+save_name)
model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=3, callbacks=[es, mcp])

#model.save(save_path+'keras_kaggle_bike_model_best2.h5')

#4. 평가, 예측
mse, mae=model.evaluate(x_test, y_test)

y_predict=model.predict(x_test)

r2=r2_score(y_test, y_predict)

def RMSE(y_test, y_predict): # RMSE를 함수로 구현한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt=루트 씌운다는 의미 mean_squared_error는 mse 연산

rmse = RMSE(y_test, y_predict)

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

submission.to_csv(data_path + 'submission_0125.csv')

#R2(결정계수)란 정확도를 의미
print('mse :', mse)
print('mae :', mae)
print("R2 :", r2)
print("RMSE :", rmse)

#random_state=300
# mse : 2727.793701171875
# mae : 36.13483428955078
# R2 : 0.9172844938738727
# RMSE : 52.22828452825661