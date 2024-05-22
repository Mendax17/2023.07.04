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

data_path='c:/Users/eagle/Downloads/bitcamp/AI/_data/cat_dog/'
save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

xy_train=train_datagen.flow_from_directory(data_path+'train_1/', target_size=(100, 100), batch_size=25000, class_mode='binary', color_mode='rgb', shuffle=True) # 폴더로 부터 데이터를 받아오는 함수
# categorical 을 하면 y값이 one-hot encoding 되서 나온다.
xy_test=train_datagen.flow_from_directory(data_path+'test/', target_size=(100, 100), batch_size=12500, class_mode='binary', color_mode='rgb', shuffle=True)

# print(xy_test)
# # <keras.preprocessing.image.DirectoryIterator object at 0x0000015EC4A803A0>

# print(xy_train[0][0].shape) # (25000, 150, 150, 3)
# print(xy_train[0][1].shape) # (25000,)
# print(xy_test[0][0].shape) # (12500, 150, 150, 3)

np.save(data_path+'cat_dog_x_train.npy', arr=xy_train[0][0]) # 큰 이미지 데이터들을 numpy 데이터화 시키는 함수이다. # batch_size=10000으로 통배치 형태로 만들어 주었기 때문에 xy_train[0][0]에 모든 이미지 데이터가 들어있다.
np.save(data_path+'cat_dog_y_train.npy', arr=xy_train[0][1]) # 통배치 형식이기 때문에 xy_train[0][1]에 모든 이미지 파일의 폴더 위치 정보가 담겨있다.

np.save(data_path+'cat_dog_x_test.npy', arr=xy_test[0][0])

# np.save(data_path+'ffbrain_y_train.npy', arr=xy_train[0]) # 이런형태는 튜플 형태이기 때문에 작동하지 않는다 따라서 xy_train[0][0] 에 .append 를 이용해서 xy_train[0][1]를 이어주면 된다.(?) 안됨