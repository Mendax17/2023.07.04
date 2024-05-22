import tensorflow as tf
import numpy as np
import pandas as pd # 데이터 조작 및 분석 API
import seaborn as sns # 데이터 시각화 API
import matplotlib.pyplot as plt # 시각화 API
import datetime # 날짜, 시간을 가져오는 API
import time # 시간을 재기위한 API
import os # Operating System의 약자로서 운영체제에서 제공되는 여러 기능을 파이썬에서 수행할 수 있게 해줍니다. 파일이나 디렉토리 조작이 가능하고, 파일의 목록이나 path를 얻을 수 있거나, 새로운 파일 혹은 디렉토리를 작성하는 것도 가능합니다.
import glob # glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다.
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, Conv2D, Flatten, SimpleRNN, LSTM, GRU, MaxPool1D, MaxPool2D, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Bidirectional, concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # ReduceLROnPlateau 는 n번 동안 갱신이 없으면 멈추는게 아니라 러닝 레이트를 설정한 비율만큼 줄여주는 기능
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator # 데이터 전처리
from sklearn import datasets #print(datasets.get_data_home()) # 다운받은 datasets의 위치를 표시해준다.
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder # 데이터 전처리
from tensorflow.keras.datasets import fashion_mnist # 10가지의 옷 종류를 분류하는 방법

save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

(x_train, y_train), (x_test, y_test)=fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28 ,28) (60000, )
# print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000, )

argument_size=40000 # 1장당 몇장으로 증폭할것인지

randidx=np.random.randint(x_train.shape[0], size=argument_size) # x_train.shape[0](6만개) 사이 숫자중에서 size=argument_size 만큼을 추출
#                           60000
print(randidx) # [ 5830  9380 34113 ... 15490 56417 40913] # 1 ~ 60000 사이숫자 4만개를 뽑은것
print(len(randidx)) # 40000

x_argument=x_train[randidx].copy()
y_argument=y_train[randidx].copy()

x_argument=x_argument.reshape(-1, 28, 28, 1)

print(x_argument.shape, y_argument.shape) # (40000, 28, 28) (40000, ) # 아직 까지 증폭X 6만개의 원본 데이터중에서 4만개를 골라낸것

train_datagen=ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5,zoom_range=1.2, shear_range=0.7, fill_mode='nearest')

x_argumented=train_datagen.flow(x_argument, y_argument, batch_size=argument_size, shuffle=True)

print(x_argumented[0][0].shape) # (40000, 28, 28, 1)
print(x_argumented[0][1].shape) # (40000, )

x_train=x_train.reshape(60000, 28, 28, 1) # 기존 6만개의 데이터에 증폭된 4만개를 더할것이다.

x_train=np.concatenate((x_train, x_argumented[0][0]))
y_train=np.concatenate((y_train, x_argumented[0][1]))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)

# 4만개의 추출한 데이터를 이미지 제너러 레이터 flow로 엮는 이미지 , flow from directory는 폴더 내 데이터를 수치화 하는것 , flow는 이미 수치화 되어 있는 애들을 데려오는것