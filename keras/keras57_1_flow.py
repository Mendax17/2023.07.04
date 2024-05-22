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
argument_size=100 # 1장당 몇장으로 증폭할것인지

train_datagen=ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1, rotation_range=5,zoom_range=1.2, shear_range=0.7, fill_mode='nearest')

# print(x_train[0].shape) # (28, 28) # 1개의 흑백 이미지
# print(x_train[0].reshape(28*28).shape) # (784, ) # 흑백 이미지를 1차원으로 변경
# print(np.tile(x_train[0].reshape(28*28, ), argument_size).shape) # (78400, ) # 1차원 흑백 이미지 1개를 * 100개를 만듬
# print(np.tile(x_train[0].reshape(28*28, ), argument_size).reshape(-1, 28, 28, 1).shape) # (100, 28, 28, 1) # 1차원 흑백 이미지 100장을 -> 4차원 이미지 100장으로 변환

x_data=train_datagen.flow( # 1개 데이터를 100개로 늘린것이다.
    np.tile(x_train[0].reshape(28*28, ), argument_size).reshape(-1, 28, 28, 1), # x 데이터 증폭
    np.zeros(argument_size), # y 데이터 증폭 # np.zeros(인수)는 받은 인수 만큼 0 으로 채워진 array를 생성합니다. # np.zeros(100)=(100,)
    batch_size=argument_size,
    shuffle=True)

print(x_data) # <keras.preprocessing.image.NumpyArrayIterator object at 0x000001BB25BE50D0>
print(type(x_data[0])) # <class 'tuple'>
print(x_data[0][0].shape) # (100, 28, 28, 1)
print(x_data[0][1].shape) # (100, ) # 모두 똑같은 신발이므로 0 이 100개인 리스트

plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
plt.show()