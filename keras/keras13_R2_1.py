import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # 그림그리는 라이브러리

#1. 데이터
x = np.array(range(1,21)) #(20, )
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) #(20, )

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7, random_state=0) # 행렬 형태의 데이터도 train_test_split 함수로 나누는게 가능하다. # randon_state는 랜덤 난수를 고정시킨다.

#2. 모델구성
model=Sequential()
model.add(Dense(15, input_dim=1)) #x의 덩어리 갯수 #행렬에서 열의 갯수와 같다 #열의 갯수가 우선된다 #열=컬럼,피처,특성 을 의미 # ex)환율,금리,물가지수 이런 요소들이 열
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(14))
model.add(Dense(15))
model.add(Dense(14))
model.add(Dense(13))
model.add(Dense(12))
model.add(Dense(11))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # loss 방식은 누적이기 때문에 그 다음 가중치 갱신에 영향을 미친다. # metrics는 훈련에 영향을 주지 않고 참고 자료로 볼 수 있다.
# 해당 loss 선택지중에서 mse는 없기 때문에 loss='rmse' 는 쓸 수 없다는 error 발생
# rmse를 사용할려면 rmse 함수를 구현하여 mse를 루트처리하는 함수로 구현해야 한다.
model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x_test, y_test) # evaluate도 batch_size가 존재함 default 32 이므로 1/1 로 나온다. # metrics=['mae']를 넣었다면 loss 반환도 2가지를 보여준다.
print('loss : ', loss)

y_predict = model.predict(x_test) # 해당 부분에서는 이미 최적의 가중치가 생성되어 있다.

print("test",y_test)
print("predict",y_predict)

from sklearn.metrics import mean_squared_error,r2_score # 2개 import # mse를 구현하는 함수 # R2(결정계수)

def RMSE(y_test, y_predict): # RMSE를 함수로 구현한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt=루트 씌운다는 의미 mean_squared_error는 mse 연산

print("RMSE : ", RMSE(y_test, y_predict))

#R2(결정계수)란 정확도를 의미
r2=r2_score(y_test,y_predict)
print("R2 : ",r2)