import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import matplotlib.pyplot as plt # 시각화 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.datasets import load_digits # 숫자 이미지 인식 데이터셋

#1. 데이터
dataset=load_digits()
x=dataset.data # (1797, 64) # 4가지 칼럼(input_dim=4)으로 결과 y(output_dim=1)는 Iris-Setosa, Iris-Versicolour, Iris-Virginica (0, 1, 2) 3가지로 구분한다.
y=dataset.target # (1797, ) # [0 ~ 0 1 ~ 1 2 ~ 2 3 ~ 3 4 ~ 4 5 ~ 5 6 ~ 6 7 ~ 7 8 ~ 8 9 ~ 9]
# x=dataset['data']
# y=dataset['target']
print(dataset.feature_names)
print(np.unique(y, return_counts=True)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 내부에 어떤 종류의 값들이 있는지 집합으로 확인 가능하다. 분류임을 확인할 수 있음
# return_counts를 True로 했을때 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180])

# plt.gray()
# plt.matshow(dataset.images[5])
# plt.show()

# One_hot_Encoding 방법1
from tensorflow.keras.utils import to_categorical # tensorflow에서 제공하는 데이터를 one hot encoding 해주는 기능이다.
y=to_categorical(y) # (1797, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y, random_state=10)
# stratify(통계적인) 옵션을 y로 주게되면 train, test 값을 같은 비율로 해서 나눠준다. 또한 stratify는 분류형에서만 사용가능하며 회귀형에서는 Error가 발생한다.
# 분류에서는 한쪽 데이터가 너무 많거나 적으면 문제가 발생한다. 만약 적은 데이터양에서 train과 test 한쪽으로 밀집되게 잘라버린다면 운없게 적은 데이터쪽에서 모두 같은 값을 가질 수도 있다.

#2. 모델구성
model=Sequential()
model.add(Dense(64, activation='relu', input_shape=(64,))) # 훈련되고 activation이 적용되는 원리
model.add(Dense(128, activation='sigmoid')) # one hot encoding 을 거쳐 0, 1, 2 가 모두 0 또는 1 로 되었으므로 sigmoid 사용이 가능하다.
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='linear'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 이진분류는 0 또는 1로만 출력 되기 때문에 몇개가 맞았는지 accuracy로 표현가능하다.
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1) # 이진분류도 accuracy보다 보편적으로 val_loss를 지표로 하는게 성능이 더 좋다.
hist=model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.2, callbacks=[earlyStopping], verbose=1)

#4. 평가, 예측
loss, accuracy=model.evaluate(x_test, y_test)
print('(loss :', loss, end=')')
print('(accuracy :', accuracy, ')')

y_predict=model.predict(x_test) # [3.7969347e-10 4.6464029e-05 9.9995351e-01] # 칼럼은 3개로 분리된 상태지만 값들은 [0, 0, 1]이 아닌 softmax에 의해 각 항들의 합이 1인 형태를 띄고 있음
# 따라서 이를 완전한 one hot encoding 형태인 [0, 0, 1]로 만들어 준다음 최종적으로 2 로 변환하는 작업을 거쳐야함
#print(y_predict.shape) # (30, 3)

y_predict=np.argmax(y_predict, axis=1) # axis=1 일때는 행을 비교해서 그 행에서 softmax 형태의 확률을 확인하여 다시 one hot encoding 상태전으로 되돌려준다.
y_test=np.argmax(y_test, axis=1) # [3.2756470e-10 3.9219194e-05 9.9996078e-01] --> [0, 0, 1] --> [2] --> 결과적으로 [2 1 2 0 2]

print(y_predict[:10])
print(y_test[:10])

acc=accuracy_score(y_test, y_predict)
print('accuarcy_score :', acc)