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
from sklearn.datasets import load_iris # 꽃에 대한 4가지 칼럼 정보를 보고 꽃을 맞추는 데이터셋

#1. 데이터
dataset=load_iris()
x=dataset.data
y=dataset.target

# One_hot_Encoding 방법1
from tensorflow.keras.utils import to_categorical # tensorflow에서 제공하는 데이터를 one hot encoding 해주는 기능이다.
y=to_categorical(y) # (150, 3)

# One_hot_Encoding 방법2
# y=tf.one_hot(y, 3) # (150, 3) # One-hot-Encoding

print(dataset.DESCR) # pands.describe() / .info()
# ============== ==== ==== ======= ===== ====================
#                 Min  Max   Mean    SD   Class Correlation
# ============== ==== ==== ======= ===== ====================
# sepal length:   4.3  7.9   5.84   0.83    0.7826
# sepal width:    2.0  4.4   3.05   0.43   -0.4194
# petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)     # Class Correlation가 높은 데이터는 비슷한 데이터라는 뜻으로 둘중 하나가 없어도 크게 차이가 없을 수 도 없다는 의미이다.
# petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)     # 무조건 데이터가 많다고 좋은것은 아니다. 오히려 연관성이 떨어지는 데이터는 모델 성능 자체를 낮출수도 있다.
# ============== ==== ==== ======= ===== ====================
print(dataset.feature_names) # pands.columns

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
input1=Input(shape=(4, ))
dense1=Dense(32, activation='relu')(input1) # layer 마다 input layer를 정의해줘야한다.
dense2=Dense(64, activation='sigmoid')(dense1)
dense3=Dense(32, activation='relu')(dense2)
dense4=Dense(32, activation='relu')(dense3)
dense5=Dense(16, activation='relu')(dense4)
dense6=Dense(8, activation='linear')(dense5)
output1=Dense(3, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1) # 시작과 끝모델을 직접 지정해준다.
# 수치화의 주의할점은 0, 1, 2 라는 label로 구별할때 1이 2배해서 2가 된다고 해도 이는 label로 구별한것이기 뿐이므로 이를 같게 보면 안된다는 뜻이다.
# One-Hot Encoding : 0, 1, 2 라는 labeling을 할때 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 이진분류는 0 또는 1로만 출력 되기 때문에 몇개가 맞았는지 accuracy로 표현가능하다.
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1) # 이진분류도 accuracy보다 보편적으로 val_loss를 지표로 하는게 성능이 더 좋다.
hist=model.fit(x_train, y_train, epochs=500, batch_size=8, validation_data=(x_val, y_val), verbose=1, callbacks=[earlyStopping]) # model.fit의 어떤 반환값을 hist에 넣는다.

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('(loss :', loss, end=')')
print('(accuracy :', accuracy, ')')

from sklearn.metrics import accuracy_score
y_predict=model.predict(x_test) # [3.7969347e-10 4.6464029e-05 9.9995351e-01] # 칼럼은 3개로 분리된 상태지만 값들은 [0, 0, 1]이 아닌 softmax에 의해 각 항들의 합이 1인 형태를 띄고 있음
# 따라서 이를 완전한 one hot encoding 형태인 [0, 0, 1]로 만들어 준다음 최종적으로 2 로 변환하는 작업을 거쳐야함
#print(y_predict.shape) # (30, 3)

y_predict=np.argmax(y_predict, axis=1) # axis=1 일때는 행을 비교해서 그 행에서 softmax 형태의 확률을 확인하여 다시 one hot encoding 상태전으로 되돌려준다.
y_test=np.argmax(y_test, axis=1) # [3.2756470e-10 3.9219194e-05 9.9996078e-01] --> [0, 0, 1] --> [2] --> 결과적으로 [2 1 2 0 2] # y_test는 one hot encoding을 해주지 않았을때는 argmax를 돌리면 Error 발생

# print(y_predict[:5])
# print(y_test[:5])

acc=accuracy_score(y_test, y_predict)
print('accuarcy_score :', acc)

# acc : 0.9444