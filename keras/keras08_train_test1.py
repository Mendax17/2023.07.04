import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
#x = np.array([1,2,3,4,5,6,7,8,9,10]) #(10, )
#y = np.array(range(10)) #(10, )
# 이 경우는 y=wx+b --> y=x+1 (w=1, b=1) 형태이다.

# 데이터를 train data 와 test data 2개로 분류하여 사용한다.
x_train=np.array([1,2,3,4,5,6,7]) #(7, )
x_test=np.array([8,9,10]) #(3, )
# 이 경우 특성이 똑같이 1이기 때문에 트레이닝이 정상 구동한다. (열만 동일하면 정상 동작된다.)
y_train=np.array(range(7)) #(7, )
y_test=np.array(range(7,10)) #(3, )

#2. 모델구성
model=Sequential()
model.add(Dense(10, input_dim=1)) #x의 덩어리 갯수 #행렬에서 열의 갯수와 같다 #열의 갯수가 우선된다 #열=컬럼,피처,특성 을 의미 # ex)환율,금리,물가지수 이런 요소들이 열
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
# Dense의 모양이 퍼졌다가 줄어들면(Dense(1)로 갈 경우) 줄어들때 상당수의 데이터가 손실되므로 퍼졌다가 서서히 줄어드는 형태가 많다. 줄어들었다가 다시 퍼지는 형태 X (Dense(1)-->Dense(10))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x_test, y_test) # evaluate도 batch_size가 존재함 default 32 이므로 1/1 로 나온다.
print('loss : ', loss)
result = model.predict([11]) #대괄호 2개
print('예측값 : ',result) #[11]