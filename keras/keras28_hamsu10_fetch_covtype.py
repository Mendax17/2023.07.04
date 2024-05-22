import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model # Model은 함수형 모델이다.
from tensorflow.keras.layers import Dense, Input # 함수형 모델은 input layer를 정해줘야한다.
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from sklearn.datasets import fetch_covtype # 토지조사 (회귀 분석용)

from sklearn import datasets
#print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.

#1. 데이터
dataset=fetch_covtype()
x=dataset.data # (581012, 54)
y=dataset.target # (581012, )
# x=dataset['data']
# y=dataset['target']
print(x.shape, y.shape)
print(dataset.feature_names)
# ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4', 'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24', 'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39']
print(np.unique(y, return_counts=True))
# return_counts를 True로 했을때 array([1, 2, 3, 4, 5, 6, 7]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180])


########################### One_hot_Encoding 방법1 (tensorflow의 to_categorical) ###########################
# from tensorflow.keras.utils import to_categorical # tensorflow에서 제공하는 데이터를 one hot encoding 해주는 기능이다.
# #y=[index-1 for index in y]
# y=to_categorical(y) # (581012, 8) # to_categorical는 무조건 0부터 시작하기 때문에 현재처럼 1부터 시작할때는 자동으로 0부분을 생성하기 때문에 8개의 칼럼으로 잡힌다. 따라서 삭제해줘야함
# print(y[:10])
# print(np.unique(y[:,0], return_counts=True)) # 0번째 컬럼의 유니크한걸 보여달라. # (array([0.], dtype=float32), array([581012], dtype=int64)) # 0 밖에 없다는걸 확인할 수 있다.
# print(np.unique(y[:,1], return_counts=True)) # 1번째 컬럼의 유니크한걸 보여달라. # (array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64)) # 0, 1 2가지가 있는걸 확인할 수 있다.
# # 따라서 0번째 칼럼이 필요없이 생성되었다는걸 확인할 수 있다.
# y=np.delete(y, 0, axis = 1) # numpy 자료형이므로 numpy 삭제 방법을 사용 # 2번째에 들어갈 변수는 몇번째 인덱스를 삭제할것인지 결정 # axis=0 일때는 세로 방향(행) 삭제 axis=1 일때는 가로 방향 삭제
# print(np.unique(y[:,0], return_counts=True)) # (array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64)) # 0번째 행이 제대로 지워진걸 확인할 수 있음
############################################################################################################

########################### One_hot_Encoding 방법2 (pandas의 get_dummies) ###################################
# y=pd.get_dummies(y, columns=['example']) # get_dummies의 자료형을 확인하고 받아들일 수 있는지
# print(y) # pands의 데이터 형태는 index와 header가 자동으로 생성된다.
# #         1  2  3  4  5  6  7
# # 0       0  0  0  0  1  0  0
# # 1       0  0  0  0  1  0  0
# # 2       0  1  0  0  0  0  0
# # 3       0  1  0  0  0  0  0
# # 4       0  0  0  0  1  0  0
# print(y.shape) # (581012, 7) # 배열 형태는 연산 조건과 맞다. 훈련은 배열 형태만 맞다면 pandas형 데이터든 numpy형 데이터든 정상적으로 진행된다.
# print(type(y)) # <class 'pandas.core.frame.DataFrame'> # 연산은 정상적으로 진행되지만 numpy.argmax 함수에서는 numpy형 데이터만 받을 수 있으므로 Error가 발생한다.
# y=y.values # numpy 자료형으로 변경하는 방법1
# #y=y.to_numpy() # numpy 자료형으로 변경하는 방법2
# print(y)
# #[[0 0 0 ... 1 0 0][0 0 0 ... 1 0 0] ... [0 0 1 ... 0 0 0][0 0 1 ... 0 0 0]]
# print(type(y)) # <class 'numpy.ndarray'>
############################################################################################################

########################### One_hot_Encoding 방법3 (sklearn의 OneHotEncoder) ################################
from sklearn.preprocessing import OneHotEncoder # preprocessing이 전처리라는 의미이다.
ohe=OneHotEncoder()
# 방법1
y=y.reshape(581012, 1) # 자료의 배열 형태를 바꿀때 1. 내용물이 바뀌면 안되고 2. 순서가 바뀌면 안된다. 이 2가지만 지키면 형태는 변경은 무관
# reshape(index1, index2) 배열을 바꿔주는 함수 index1는 행의 갯수 index2는 열의 갯수를 조절하는 함수입니다. -1은 다른 행 또는 열 값에 맞춰 자동으로 맞춰준다는 의미입니다.
print(y.shape) # (581012, 1)
ohe.fit(y) # ValueError: Expected 2D array, got 1D array instead # one hot encoding은 2차원 배열만 받는다. # y를 OneHotEncoder에 넣었다는 의미이다.
y=ohe.transform(y) # fit을 거치면서 훈련시킨 가중치를 ohe에 저장하고 transform은 들어온 데이터들 그 가중치에 맞는 데이터 형식으로 바꿔준다.
print(y[:10])
# (0, 4)        1.0 # (0, 4) 좌표가 1 이라는 의미
# (1, 4)        1.0 # (1, 4) 좌표가 1 이라는 의미
# (2, 1)        1.0
# (3, 1)        1.0
# (4, 4)        1.0
# (5, 1)        1.0
# (6, 4)        1.0
# (7, 4)        1.0
# (8, 4)        1.0
# (9, 4)        1.0
print(y.shape) # (581012, 7)
print(type(y)) # <class 'scipy.sparse._csr.csr_matrix'>
# TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array. # .toarray()를 사용하여 데이터 형태를 바꿔줘야함
y=y.toarray() # (581012, 7) # 열정렬을 행정렬로 바꿔주는 명령어
print(type(y)) # <class 'numpy.ndarray'>
#y=ohe.fit_transform(y.reshape(-1, 1)) # fit 과 transform 을 한번에 처리해주는 명령어이다.
#############################################################################################################

x_train, x_input, y_train, y_input = train_test_split(x, y, train_size=0.7, stratify=y,random_state=333)
x_test, x_val, y_test, y_val = train_test_split(x_input, y_input, train_size=0.8, random_state=333)

#scaler=StandardScaler()
scaler = MinMaxScaler() # MinMaxScaler를 scaler라는 이름으로 정의한다. # 항상 좋은것 X 적절한 사용 필요
scaler.fit(x_train) # x값은 변하지 않고 x 데이터를 활용하여 MinMaxScaler의 전처리 조건에 맞는 가중치를 생성한다는 의미
#x_train=scaler.fit_transform(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_val=scaler.transform(x_val)

#2. 모델구성
input1=Input(shape=(54, ))
dense1=Dense(64, activation='linear')(input1) # layer 마다 input layer를 정의해줘야한다.
dense2=Dense(128, activation='sigmoid')(dense1)
dense3=Dense(64, activation='relu')(dense2)
dense4=Dense(64, activation='relu')(dense3)
dense5=Dense(32, activation='relu')(dense4)
dense6=Dense(16, activation='relu')(dense5)
dense7=Dense(8, activation='relu')(dense6)
output1=Dense(7, activation='softmax')(dense7)
model = Model(inputs=input1, outputs=output1) # 시작과 끝모델을 직접 지정해준다.

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 이진분류는 0 또는 1로만 출력 되기 때문에 몇개가 맞았는지 accuracy로 표현가능하다.
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1) # 이진분류도 accuracy보다 보편적으로 val_loss를 지표로 하는게 성능이 더 좋다.
hist=model.fit(x_train, y_train, epochs=500, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[earlyStopping]) # model.fit의 어떤 반환값을 hist에 넣는다.

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('(loss :', loss, end=')')
print('(accuracy :', accuracy, ')')

y_predict=model.predict(x_test) # [3.7969347e-10 4.6464029e-05 9.9995351e-01] # 칼럼은 7개로 분리된 상태지만 값들은 [0, 0, 0, 0, 0, 1, 0]이 아닌 softmax에 의해 각 항들의 합이 1인 형태를 띄고 있음
# 따라서 이를 완전한 one hot encoding 형태인 [0, 0, 0, 0, 0, 1, 0]로 만들어 준다음 최종적으로 6 로 변환하는 작업을 거쳐야함

#print(y_predict) # (174304, 7)
# [0.28041425 0.60519767 0.04322869 ... 0.01788932 0.02289666 0.02679514]
#print(y_test) # (174304, 7)
#         1  2  3  4  5  6  7
# 497776  1  0  0  0  0  0  0
# 568849  0  1  0  0  0  0  0
# ...    .. .. .. .. .. .. ..
# 408359  0  0  1  0  0  0  0
# 496287  1  0  0  0  0  0  0
#y_test=y_test.values.tolist() # dataframe 형태 리스트 변환
y_predict=np.argmax(y_predict, axis=1) # numpy 자료형만 받을수 있다. # axis=1 일때는 행을 비교해서 그 행에서 softmax 형태의 확률을 확인하여 다시 one hot encoding 상태전으로 되돌려준다.
y_test=np.argmax(y_test, axis=1) # [3.2756470e-10 3.9219194e-05 9.9996078e-01] --> [0, 0, 1] --> [2] --> 결과적으로 [2 1 2 0 2]

print(y_predict[:10])
print(y_test[:10])

acc=accuracy_score(y_test, y_predict)
print('accuarcy_score :', acc)

# acc : 0.8955