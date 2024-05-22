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

#1. 데이터
# from glob import glob
# train_data=glob(data_path+'train/*.jpg')
# import glob # 이렇게 2가지 방법 있음
train_data=glob.glob(data_path+'train/*.jpg') # <class 'list'> # glob() 함수는 인자로 받은 패턴과 이름이 일치하는 모든 파일과 디렉터리의 경로를 리스트로 반환한다.
train_labels=[i.strip(data_path+'train/')[1:4] for i in train_data] # '\\cat' 또는 '\\dog' 'cat' 또는 'dog'
train_df = pd.DataFrame({'filename': train_data, 'class': train_labels}) # [25000 rows x 2 columns]

test_data=glob.glob(data_path+'test/*.jpg')
test_labels=[i.strip(data_path+'test/')[1:4] for i in test_data] # '\\cat' 또는 '\\dog' 'cat' 또는 'dog'
test_df = pd.DataFrame({'filename': test_data, 'class': test_labels}) # [25000 rows x 2 columns]

submission=pd.read_csv(data_path + 'sampleSubmission.csv', index_col=0)

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_df, val_df = train_test_split(train_df, train_size=0.8, random_state=10)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(train_df.shape, val_df.shape) # (20000, 2) (5000, 2)

file_names = os.listdir(data_path+'test/')
test_df = pd.DataFrame({'filename': file_names})
print("shape:", test_df.shape)
test_df.head()

train_generator=train_datagen.flow_from_dataframe(dataframe=train_df, x_col='filename', y_col='class', target_size=(200, 200), batch_size=4, class_mode='binary')
val_generator=train_datagen.flow_from_dataframe(dataframe=val_df, x_col='filename', y_col='class', target_size=(200, 200), batch_size=4, class_mode='binary')

test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, directory=data_path+'test/', x_col='filename', y_col=None, class_mode=None, target_size=(300, 300), batch_size=4)
# xy_train=train_datagen.flow_from_directory(data_path+'train/', target_size=(300, 300), batch_size=32, class_mode='binary', shuffle=True)
# xy_test=train_datagen.flow_from_directory(data_path+'test/', target_size=(300, 300), batch_size=32, class_mode='binary', shuffle=True)

# print(xy_train[cat])
# # <keras.preprocessing.image.DirectoryIterator object at 0x0000015EC4A803A0>

# print(xy_train[0])
# print(xy_train[0][0])
# print(xy_train[0][1]) # categorical 을 하면 y값이 one-hot encoding 되서 나온다.
# print(xy_train[0][0].shape) # (10, 200, 200, 1)
# print(xy_train[0][1].shape) # (10, 2) # one-hot encoding 되었기 때문에 (10, 1) -> (10, 2)가 된다.

# np.save(data_path+'cat_dog_x_train.npy', arr=xy_train[0][0]) # 큰 이미지 데이터들을 numpy 데이터화 시키는 함수이다. # batch_size=10000으로 통배치 형태로 만들어 주었기 때문에 xy_train[0][0]에 모든 이미지 데이터가 들어있다.
# np.save(data_path+'cat_dog_y_train.npy', arr=xy_train[0][1]) # 통배치 형식이기 때문에 xy_train[0][1]에 모든 이미지 파일의 폴더 위치 정보가 담겨있다.

# np.save(data_path+'cat_dog_x_test.npy', arr=xy_test[0][0])
# np.save(data_path+'cat_dog_y_test.npy', arr=xy_test[0][1])

#2. 모델
model=Sequential()
model.add(Conv2D(32, 3, input_shape=(200, 200, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
date_now=datetime.datetime.now()
date_now=date_now.strftime("%m%d_%H%M")
performance_info='({val_loss:.4f})'
save_name=date_now+performance_info+'.hdf5'

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es=EarlyStopping(monitor='val_loss', mode='min', patience=1, verbose=1)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=save_mcp_path+'k55_3_cat_dog_'+save_name)
hist=model.fit(train_generator, epochs=1, batch_size=16, validation_data=val_generator, verbose=1, callbacks=[es])#, mcp])

loss=hist.history['loss'] # fit에서 훈련시킨 모든 epoch당 loss를 list 형태로 가지고 있다.
accuracy=hist.history['acc']
val_acc=hist.history['val_acc']
val_loss=hist.history['val_loss']

# 제출할 데이터
y_submit=model.predict(test_generator)
print(y_submit) # y_submit 에서도 nan이 나오는것으로 결측치가 존재한다는것을 알수있다. # test_csv 에도 결측치가 존재한다는 의미(단, 제출 자료이기 때문에 결측치 삭제 X)
# 제출하기 위해 y_submit 을 submission.csv 의 count 부분에 넣어주면 된다.
print(y_submit.shape) # (1459, 10)
print(submission.shape) # (715, 1)

# .to_csv()를 사용해서 submission_0105.csv를 완성하시오.
#pd.DataFrame(y_submit).to_csv(path_or_buf=path+'submission_0105.csv', index_label=['id'], header=['count']) # 이 함수는 index 를 순차적으로 새로 설정하기 때문에 원본과 다르다.
submission['label'] = y_submit # submission의 'count'라는 컬럼에 y_submit 값을 집어넣는다.
print(submission)

submission.to_csv(data_path + 'submission_0131.csv')

print('loss :', loss[-1])
print('accuracy :', accuracy[-1])
print('val_acc :', val_acc[-1])
print('val_loss :', val_loss[-1])

# loss : 0.7021271586418152
# accuracy : 0.5030500292778015
# val_acc : 0.4903999865055084
# val_loss : 0.6936502456665039