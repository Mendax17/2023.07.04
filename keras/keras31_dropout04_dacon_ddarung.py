import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model # Model은 함수형 모델이다.
from tensorflow.keras.layers import Dense, Input, Dropout # Dropout 히든 레이어에서 일부를 연산하지 않는 기능
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

data_path='c:/Users/eagle/Downloads/bitcamp/_data/ddarung/'
save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

#1. 데이터
# 만약 파일을 불러올때 인덱스 컬럼을 지정해주지 않으면 자동으로 생성된다.
train_csv=pd.read_csv(data_path + 'train.csv', index_col=0) # 글자열 + 글자열 하면 합치는게 가능하기 때문에 중복되는 주소를 path에 str로 넣어서 표시한다. # index_col는 몇번째 행을 index로 보고 데이터 칼럼에서 뺄건지를 지정하는 명령어
#train_csv=pd.read_csv('./_data/ddarung/train.csv', index_col=0)
test_csv=pd.read_csv(data_path + 'test.csv', index_col=0) # [715 rows x 9 columns]
submission=pd.read_csv(data_path + 'submission.csv', index_col=0)

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

x=train_csv.drop(['count'], axis=1) # drop은 pandas에서 특정 칼럼을 제거 하는 명령어이다. # count라는 칼럼을 제거
y=train_csv['count']

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, random_state=60)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=60)

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

#2. 모델구성(함수형) # 함수형은 모델을 다 구성하고 마지막에 함수형을 정의한다.
input1=Input(shape=(9, ))
dense1=Dense(32, activation='linear')(input1) # layer 마다 input layer를 정의해줘야한다.
drop1=Dropout(0.1)(dense1)
dense2=Dense(64, activation='relu')(drop1)
drop2=Dropout(0.1)(dense2)
dense3=Dense(32, activation='relu')(drop2)
drop3=Dropout(0.1)(dense3)
dense4=Dense(16, activation='relu')(drop3)
dense5=Dense(8, activation='relu')(dense4)
output1=Dense(1, activation='linear')(dense5)
model = Model(inputs=input1, outputs=output1) # 시작과 끝모델을 직접 지정해준다.

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)#, restore_best_weights=False)

date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
#performance_info='({val_loss:.2f}-{epoch:04d})' # 04d 는 정수 4자리까지 표시하겠다는 의미이다. # .2f는 소수점 2번째 자리까지 표시하라는 의미이다. # 0037-0.0048.hdf5 # ModelCheckpoint에서 해당 값들을 채워준다.
performance_info='({val_loss:.2f})'
save_name=date_now+performance_info+'.hdf5'

mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k31_'+save_name)#filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
# ModelCheckpoin의 verbose=1 일때 # Epoch 00025: val_loss improved from 10.33085 to 9.83417, saving model to c:/Users/eagle/Downloads/bitcamp/_save/MCP\keras30_ModelCheckPoint1.hdf5 # 어떤 epoch에서 val_loss가 개선된 값이 나와 저장했는지에 대한 정보를 알 수 있다.
hist=model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[es, mcp]) # callbacks=[es, mcp] 2개 이상은 [] list로 표현

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

submission.to_csv(data_path + 'submission_0112.csv')

#R2(결정계수)란 정확도를 의미
print('mse : ', mse)
print('mae : ', mae)
print("R2 :", r2)
print("RMSE : ", rmse)

# RMSE : 39.45