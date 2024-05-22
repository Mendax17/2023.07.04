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
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse', 'accuracy', 'acc']) # loss 방식은 누적이기 때문에 그 다음 가중치 갱신에 영향을 미친다. # metrics는 훈련에 영향을 주지 않고 참고 자료로 볼 수 있다.
# 'accuracy' = 'acc' 는 줄임말로 동작 기능은 같다. 단 여기서는 쓸 수 없는 지표
model.fit(x_train,y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x_test, y_test) # evaluate도 batch_size가 존재함 default 32 이므로 1/1 로 나온다.
print('loss : ', loss)

# mae(평균 절대 오차) : 1.77427518 # 큰 범위차에서 좋음
# mse(평균 제곱 오차) : 17.3385028 # 작은 범위차에서 좋음 #값을 제곱하기 때문에 절댓값이 1미만인 값은 더 작아지고, 1보다 큰 값은 더 커지는 왜곡이 발생할 수 있다.