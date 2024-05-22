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
from tensorflow.keras.datasets import mnist # tensorflow 데이터셋

(x_train, y_train), (x_test, y_test)=mnist.load_data() # 이미 train과 test가 분리되어 있는 데이터셋
print(x_train.shape, y_train.shape) # (60000, 28 ,28) # x는 60000만장 가로 28 세로 28 생략된 색깔 1 이렇게 생각할 수 있다. 따라서 reshape로 (28, 28, 1)로 바꿔준다.
print(x_test.shape, y_test.shape) # (10000, 28, 28)

print(x_train[0]) # 가로 28 세로 28 의 그림이고 0 검정색 배경의 5라는 하얀색 글자가 써져있을것이다.
print(y_train[0]) # 5

plt.imshow(x_train[1000], 'gray')
plt.show()