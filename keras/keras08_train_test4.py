import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)]) #(3,10)
y = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]) #(2,10)

x=x.T #(10,3)
y=y.T #(10,2)
# 일반적인 행렬 곱셈과는 다르게 특징을 무조건 열에 두고 연산 해야한다.

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0) # 행렬 형태의 데이터도 train_test_split 함수로 나누는게 가능하다.

print('x_train : ',x_train)
print('x_test : ',x_test)
print('y_train : ',y_train)
print('y_test : ',y_test)

model=Sequential()
model.add(Dense(15, input_dim=3)) #x의 덩어리 갯수 #행렬에서 열의 갯수와 같다 #열의 갯수가 우선된다 #열=컬럼,피처,특성 을 의미 # ex)환율,금리,물가지수 이런 요소들이 열
# 컬럼(관계형 데이터베이스 테이블에서 특정한 단순 자료형의 일련의 데이터값과 테이블에서의 각 열) # feature(특징)
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
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x_test, y_test) # evaluate도 batch_size가 존재함 default 32 이므로 1/1 로 나온다.
print('loss : ', loss)
results = model.predict([[9,30,210]]) #대괄호 2개
print('예측값 : ',results) #[[10],[1.4]]