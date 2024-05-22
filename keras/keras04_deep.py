import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential #순차적 진행 형식
from tensorflow.keras.layers import Dense
#한줄 라인 지우기 SHIFT + DELETE

#1. 데이터 준비
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,5,4])

#2. 모델구성
# input layer에서 output 3개 그 다음 레이어에서는 input 3개 output 4개 형식
model=Sequential() #model을 Sequential 모델로 설정한다.
model.add(Dense(9,input_dim=1))#input layer Sequential 모델에 포함(add)할것이다 위에 선택한 Dense 모델을 적용할것이고 이 모델의 input, output 관계는 input 1개 output 3개이다.
model.add(Dense(8))#hidden layer
model.add(Dense(9))#hidden layer
model.add(Dense(8))#hidden layer
model.add(Dense(8))#hidden layer
model.add(Dense(9))#hidden layer
model.add(Dense(8))#hidden layer
model.add(Dense(9))#hidden layer
model.add(Dense(1))#output layer
#node의 갯수를 늘리거나 hidden layer를 추가하는것으로 input, output, epochs를 건드리지않고 정확도를 높을수있다.

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=200) # x(input), y(output), epochs(연산횟수)

#4. 평가, 예측
results = model.predict([6])
print('6의 예측값 : ',results)