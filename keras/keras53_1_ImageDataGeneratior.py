import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Conv2D, Flatten, SimpleRNN, LSTM, GRU, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Bidirectional, concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 데이터 전처리
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리

data_path='c:/Users/eagle/Downloads/bitcamp/AI/_data/brain/'
save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

train_datagen=ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, rotation_range=5,zoom_range=1.2, shear_range=0.7, fill_mode='nearest')
# ImageDataGenerator는 동일한 이미지가 훈련을 반복할수록 발생하는 과적합을 방지하고 데이터수를 늘려주기 위해 같은 이미지를 여러 방식으로 증폭시키는 기능이다.
# 255로 나눈다는건 이미지값을 Minmaxscal 처리를 하겠다는 의미이다.(모든값이 0~1 사이값) # 수평으로 반전 시킨다. # 수직으로 반전 시킨다는 의미 # 가로 방향으로 이동시킨다 # 회전시킨다 # 1.2만큼 확대시키겠다. # 비어있는 부분을 어떤 방식으로 채울것인가

test_datagen=ImageDataGenerator(rescale=1./255) # test_data는 검증만 하는 data기 때문에 증폭시킬 필요가 없다.

xy_train=train_datagen.flow_from_directory(data_path+'train/', target_size=(200, 200), batch_size=100000, class_mode='binary', color_mode='grayscale', shuffle=True) # 폴더로 부터 데이터를 받아오는 함수
# 현재 x값과 y값이 같이 섞여있는 상태이다.
# print(xy_train[0][0]) 는 x값이고
# print(xy_train[0][1]) 는 y값이다.
# target_size는 모든 이미지를 동일한 사이즈로 맞춰준다. # batch_size=10
# Found 160 images belonging to 2 classes.
xy_test=train_datagen.flow_from_directory(data_path+'test/', target_size=(200, 200), batch_size=10, class_mode='binary', color_mode='grayscale', shuffle=True)
# Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000015EC4A803A0>

print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][0].shape) # (10, 200, 200, 1)
print(xy_train[0][1].shape)

print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> # 튜플 타입은 내부값을 바꿀수 없다.
print(type(xy_train[0][0])) # <class 'numpy.ndarray'> # x값
print(type(xy_train[0][1])) # <class 'numpy.ndarray'> # y값