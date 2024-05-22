import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Conv2D, Flatten, SimpleRNN, LSTM, GRU, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Bidirectional, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

data_path='c:/Users/eagle/Downloads/bitcamp/AI/_data/stock/'
save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

train1_csv=pd.read_csv(data_path + 'samsumg_stock.csv', encoding='cp949')#, index_col=0) # datetime을 가져오기 위해 인덱스 컬럼 지정 안해줌
train2_csv=pd.read_csv(data_path + 'amorepacific_stock.csv', encoding='cp949')#, index_col=0) # datetime을 가져오기 위해 인덱스 컬럼 지정 안해줌

print(train1_csv.shape) # (1980, 17)
print(train2_csv.shape) # (2220, 17)

train1_csv['시가'] = train1_csv['시가'].apply(lambda x: float(x.split()[0].replace(',', '')))
train1_csv['고가'] = train1_csv['고가'].apply(lambda x: float(x.split()[0].replace(',', '')))
train1_csv['저가'] = train1_csv['저가'].apply(lambda x: float(x.split()[0].replace(',', '')))
train1_csv['종가'] = train1_csv['종가'].apply(lambda x: float(x.split()[0].replace(',', '')))
train1_csv['거래량'] = train1_csv['거래량'].str.replace(pat=r'[^\w]', repl=r"", regex=True)
train1_csv['금액(백만)'] = train1_csv['금액(백만)'].str.replace(pat=r'[^\w]', repl=r"", regex=True)
train1_csv['개인'] = train1_csv['개인'].apply(lambda x: float(x.split()[0].replace(',', '')))
train1_csv['기관'] = train1_csv['기관'].apply(lambda x: float(x.split()[0].replace(',', '')))
train1_csv['외인(수량)'] = train1_csv['외인(수량)'].apply(lambda x: float(x.split()[0].replace(',', '')))
train1_csv['외국계'] = train1_csv['외국계'].apply(lambda x: float(x.split()[0].replace(',', '')))
train1_csv['프로그램'] = train1_csv['프로그램'].apply(lambda x: float(x.split()[0].replace(',', '')))

train2_csv['시가'] = train2_csv['시가'].apply(lambda x: float(x.split()[0].replace(',', '')))
train2_csv['고가'] = train2_csv['고가'].apply(lambda x: float(x.split()[0].replace(',', '')))
train2_csv['저가'] = train2_csv['저가'].apply(lambda x: float(x.split()[0].replace(',', '')))
train2_csv['종가'] = train2_csv['종가'].apply(lambda x: float(x.split()[0].replace(',', '')))
train2_csv['거래량'] = train2_csv['거래량'].str.replace(pat=r'[^\w]', repl=r"", regex=True)
train2_csv['금액(백만)'] = train2_csv['금액(백만)'].str.replace(pat=r'[^\w]', repl=r"", regex=True)
train2_csv['개인'] = train2_csv['개인'].apply(lambda x: float(x.split()[0].replace(',', '')))
train2_csv['기관'] = train2_csv['기관'].apply(lambda x: float(x.split()[0].replace(',', '')))
train2_csv['외인(수량)'] = train2_csv['외인(수량)'].apply(lambda x: float(x.split()[0].replace(',', '')))
train2_csv['외국계'] = train2_csv['외국계'].apply(lambda x: float(x.split()[0].replace(',', '')))
train2_csv['프로그램'] = train2_csv['프로그램'].apply(lambda x: float(x.split()[0].replace(',', '')))

train1_csv['일자'] = pd.to_datetime(train1_csv['일자'])
train2_csv['일자'] = pd.to_datetime(train2_csv['일자'])

train1_csv['year'] = train1_csv['일자'].apply(lambda x: x.year)
train1_csv['month'] = train1_csv['일자'].apply(lambda x: x.month)
train1_csv['day'] = train1_csv['일자'].apply(lambda x: x.day)

train2_csv['year'] = train2_csv['일자'].apply(lambda x: x.year)
train2_csv['month'] = train2_csv['일자'].apply(lambda x: x.month)
train2_csv['day'] = train2_csv['일자'].apply(lambda x: x.day)

# 결측치 제거
train1_csv=train1_csv.dropna()
train2_csv=train2_csv.dropna()

#데이터수 맞추기
train1_csv=train1_csv[:1100] # 분할 이전 가격 제거
train2_csv=train2_csv[:1100]
#train2_csv=train2_csv[:1977]

print(train1_csv.shape) # (1977, 20)
print(train2_csv.shape) # (1977, 20)

x1=train1_csv.drop(['일자', '시가', '고가', '저가', '전일비', 'Unnamed: 6', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
x2=train2_csv.drop(['일자', '시가', '고가', '저가', '전일비', 'Unnamed: 6', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
y=train1_csv['시가']

x1['등락률']=x1['종가']*(1+(x1['등락률']/100))
x2['등락률']=x2['종가']*(1+(x2['등락률']/100))

#plt.plot(train1_csv['일자'], train1_csv['시가'])
# plt.scatter(train2_csv['일자'], train2_csv['시가'])
# plt.show()

print(x1.info())
print(x2.info())
print(y.info())

# x1=pd.DataFrame(x1)
# x2=pd.DataFrame(x2)

x1=x1.astype('float')
x2=x2.astype('float')
y=y.astype('float')

print(x1.shape) # (1977, 5) # colums=[종가, 등락률, 년, 월, 일]
print(x2.shape) # (1977, 5)
print(y.shape) # (1977, )

x1_train, x1_input, x2_train, x2_input, y_train, y_input = train_test_split(x1, x2, y, train_size=0.8, shuffle=True, random_state=333)
x1_test, x1_val, x2_test, x2_val, y_test, y_val = train_test_split(x1_input, x2_input, y_input, train_size=0.8, shuffle=True, random_state=333)

# scaler = MinMaxScaler()
# x1_train=scaler.fit_transform(x1_train)
# x2_train=scaler.fit_transform(x2_train)
# x1_test=scaler.transform(x1_test)
# x2_test=scaler.transform(x2_test)
# x1_val=scaler.transform(x1_val)
# x2_val=scaler.transform(x2_val)

x1_train=x1_train.to_numpy().reshape(-1, 5, 1)
x2_train=x2_train.to_numpy().reshape(-1, 5, 1)
x1_test=x1_test.to_numpy().reshape(-1, 5, 1)
x2_test=x2_test.to_numpy().reshape(-1, 5, 1)
x1_val=x1_val.to_numpy().reshape(-1, 5, 1)
x2_val=x2_val.to_numpy().reshape(-1, 5, 1)

print(x1_train.shape, x1_test.shape) # (1581, 5) (396, 5)
print(x2_train.shape, x2_test.shape) # (1581, 5) (396, 5)
print(y_train.shape, y_test.shape) # (1581, ) (396, )

#2 모델구성(함수형)

#2-1. 모델1
input1=Input(shape=(5, 1))
dense1=LSTM(64, activation='linear', return_sequences=False)(input1)
dense2=Dense(64, activation='linear')(dense1)
dense3=Dense(128, activation='linear')(dense2)
output1=Dense(256, activation='linear')(dense3)

#2-2. 모델2
input2=Input(shape=(5, 1))
dense21=LSTM(64, activation='linear', return_sequences=False)(input2)
dense22=Dense(1, activation='linear')(dense21)
dense23=Dense(1, activation='linear')(dense22)
output2=Dense(1, activation='linear')(dense23)

# #2-1. 모델1
# input1=Input(shape=(5, 1))
# dense1=LSTM(64, activation='linear', return_sequences=True)(input1)
# drop1=Dropout(0.2)(dense1)
# dense2=LSTM(64, activation='linear', return_sequences=True)(drop1)
# drop2=Dropout(0.2)(dense2)
# dense3=LSTM(64, activation='linear', return_sequences=True)(drop2)
# drop3=Dropout(0.2)(dense3)
# dense4=LSTM(64, activation='linear', return_sequences=True)(drop3)
# output1=Dropout(0.2)(dense4)

# #2-2. 모델2
# input2=Input(shape=(5, 1))
# dense2_1=LSTM(1, activation='linear', return_sequences=True)(input2)
# drop2_1=Dropout(0.9)(dense2_1)
# dense2_2=LSTM(1, activation='linear', return_sequences=True)(drop2_1)
# drop2_2=Dropout(0.9)(dense2_2)
# dense2_3=LSTM(1, activation='linear', return_sequences=True)(drop2_2)
# drop2_3=Dropout(0.9)(dense2_3)
# dense2_4=LSTM(1, activation='linear', return_sequences=True)(drop2_3)
# output2=Dropout(0.9)(dense2_4)

#2-3. 모델병합
merge1=concatenate([output1, output2]) # 병합모델의 input은 모델1 과 모델2 의 제일 마지막 레이어가 input으로 들어온다.
merge2=Dense(256, activation='linear')(merge1)
merge3=Dense(128, activation='linear')(merge2)
merge4=Dense(64, activation='linear')(merge3)
merge5=Dense(32, activation='linear')(merge4)
merge6=Dense(16, activation='linear')(merge5)
merge7=Dense(8, activation='linear')(merge6)
last_output=Dense(1, name='last')(merge7)

# merge2=LSTM(64, activation='linear', return_sequences=True)(merge1)
# drop3_1=Dropout(0.2)(merge2)
# merge3=LSTM(64, activation='linear', return_sequences=True)(drop3_1)
# drop3_2=Dropout(0.2)(merge3)
# merge4=LSTM(64, activation='linear', return_sequences=True)(drop3_2)
# drop3_3=Dropout(0.2)(merge4)
# merge5=LSTM(64, activation='linear', return_sequences=False)(drop3_3)
# drop3_4=Dropout(0.2)(merge5)
# last_output=Dense(1, name='last')(drop3_4)

model = Model(inputs=[input1, input2], outputs=last_output) # 시작과 끝모델을 직접 지정해준다.

model.summary()

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)#, restore_best_weights=False)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k52_ensemble_hw_'+save_name)
# model.fit([x1_train, x2_train], y_train, epochs=1, batch_size=32, validation_data=([x1_val, x2_val], y_val), verbose=3, callbacks=[es, mcp])

# model.save(save_path+'keras52_ensemble_save_model.h5')
# model.save_weights(save_path+'keras52_ensemble_save_weights.h5')
# model=load_model(save_path+'keras52_ensemble_save_model.h5')
model.load_weights(save_path+'keras52_ensemble_save_weights.h5')

#4. 평가, 예측
loss_mse, loss_mae=model.evaluate([x1_test, x2_test], y_test)
print('(mse :', loss_mse, end=')')
print('(mae :', loss_mae, ')')

x1_predict=np.array([x1['종가'][0], x1['등락률'][0], 2023, 1, 30]).reshape(-1, 5, 1) # (5, )
x2_predict=np.array([x2['종가'][0], x2['등락률'][0], 2023, 1, 30]).reshape(-1, 5, 1)

result=model.predict([x1_predict, x2_predict]) # 2023.01.30
print('2023.01.30 삼성 시가 예측 결과 :', result)