import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model # Model은 함수형 모델이다.
from tensorflow.keras.layers import Dense, Input # 함수형 모델은 input layer를 정해줘야한다.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

data_path='c:/Users/eagle/Downloads/bitcamp/_data/bike/'
save_path='c:/Users/eagle/Downloads/bitcamp/_save/'

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

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, random_state=330)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=330)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)
test_csv=scaler.transform(test_csv)

print(x) # [0.00000000e+00 1.80000000e-01 6.78152493e-02 ... 2.87234043e-01 1.00000000e+00 8.96799117e-02]
print(type(x)) # <class 'numpy.ndarray'>
print("최소값 :", np.min(x)) # 0.0
print("최대값 :", np.max(x)) # 1.0

#2. 모델구성
input1=Input(shape=(12, ))
dense1=Dense(64, activation='linear')(input1) # layer 마다 input layer를 정의해줘야한다.
dense2=Dense(128, activation='sigmoid')(dense1)
dense3=Dense(64, activation='relu')(dense2)
dense4=Dense(64, activation='relu')(dense3)
dense5=Dense(32, activation='relu')(dense4)
dense6=Dense(16, activation='relu')(dense5)
dense7=Dense(8, activation='relu')(dense6)
output1=Dense(1, activation='linear')(dense7)
model = Model(inputs=input1, outputs=output1) # 시작과 끝모델을 직접 지정해준다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True, verbose=1)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_path+'MCP/keras_kaggle_bike_checkpoint_best2.hdf5')
# monitor(관측 대상), mode(최소, 최대중에서 어떤걸 찾을건지 accuracy일때는 최대 사용), patience(갱신되고 몇번 동안 갱신 안되면 반환할건지)
# restore_best_weights(earlystropping 기능이 작동 했을때 까지중 설정한 최소 or 최대 값을 저장한다 default=False), verbose(EarlyStopping이 작동했을때 값들을 보여준다)
hist=model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[es,mcp]) # model.fit의 어떤 반환값을 hist에 넣는다.

model.save(save_path+'keras_kaggle_bike_model_best2.h5')

#4. 평가, 예측
mse, mae=model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict=model.predict(x_test)

print("y_test(원래값) :", y_test)
r2=r2_score(y_test, y_predict)
print("r2 :", r2)

def RMSE(y_test, y_predict): # RMSE를 함수로 구현한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt=루트 씌운다는 의미 mean_squared_error는 mse 연산

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

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

submission.to_csv(data_path + 'submission_0112.csv')

#R2(결정계수)란 정확도를 의미
r2=r2_score(y_test,y_predict)
print("R2 : ",r2)
print("RMSE : ", rmse)

#random_state=300
# R2 : 0.9492
# RMSE : 40.89