import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model # Model은 함수형 모델이다.
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten # Conv2D 2차원 이미지 cnn 연산 1차원은 Conv1D # Flatten 차원을 내려서 펴준다.
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

save_path='c:/Users/eagle/Downloads/bitcamp/_save/'
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/_save/MCP/'

model=Sequential()
# input=(60000, 5, 5, 1) 5x5 흑백 그림이 60000개라는 의미
model.add(Conv2D(filters=10, kernel_size=(2, 2), input_shape=(10, 10, 1))) # kernel_size 쪼갤 픽셀 크기 # input_shape 은 넣어줄 이미지 크기 # (2, 2)로 쪼개니깐 (4, 4)에 필터가 10개인 상태가 되었다. # 행을 무시해서 통사적으로 (N, 4, 4, 10) 으로도 표시 가능 # N는 None이라는 의미로 데이터 갯수이므로 몇장을 넣든 의미없다는 의미 # param 2 * 2 * 10 + 10 = 50
                # (batch_size(훈련의 갯수), rows(가로), columns(세로), channels(색깔)) 가로 세로 색깔
model.add(Conv2D(5, kernel_size=(2, 2))) # (4(가로), 4(세로), 10(필터갯수))가 들어와서 (3, 3, 5)가 된다. # (N, 3, 3, 5) 으로도 표시 가능
model.add(Conv2D(7, (2, 2)))
model.add(Conv2D(6, 2))
# Conv2D의 kernel_size 는 꼭 shape 모양을 띄지 않고 단순히 2 이면 (2, 2) 3 이면 (3, 3)을 의미한다. # strides는 [kernel_size와 동일] 몇칸을 움직일 것인가
# Conv2D 갯수가 너무 많아지면 특성이 높은 애들끼리 만나 소멸된다 적당한 수치 필요
model.add(Flatten()) # 일렬로 펴주기 때문에 3 * 3 * 5 = (45, ) 가 된다. # (N, 45) 으로도 표시 가능 # 펴기만 하기 때문에 연산 X
model.add(Dense(units=10)) # 45 * 10 + 10 * 1 = 460 (450은 weight 10은 bias 따라서 460)# (N, 10)
    # input은 (batch_size, input_dim)
model.add(Dense(4, activation='relu')) # (N, 1) # 1이면 회귀로 나옴 어떤 숫자가 나옴 # 하지만 여기서는 0일때는 오바마 1일때는 트럼프 2일때는 부쉬 이렇게 구분되서 나와야되기 때문에 4 이다.

model.summary()