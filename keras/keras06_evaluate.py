import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#한줄 라인 지우기 SHIFT + DELETE

#1. 데이터 준비
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,5,4,6])

#2. 모델구성
model=Sequential() #model을 Sequential 모델로 설정한다.
model.add(Dense(10,input_dim=1))
model.add(Dense(10))#hidden layer
model.add(Dense(10))#hidden layer
model.add(Dense(10))#hidden layer
model.add(Dense(10))#hidden layer
model.add(Dense(1))#output layer

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=10, batch_size=2) #해당 부분에서 가중치가 생성된다.

#4. 평가, 예측
loss=model.evaluate(x,y) #input data, output data를 넣어 평가한다. 실제 평가데이터는 훈련시킨 데이터가 들어가면 안된다.(좋은 값이 나올 확률이 높음)
print('loss : ', loss)
results = model.predict([6])
print('6의 예측값 : ',results)
#loss 와 predict중에서는 loss 값이 가중치에 더 가까운지를 의미하기 때문에 loss를 더 우선적으로 본다.