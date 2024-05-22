import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Embedding, Flatten, Dot, Dense
from tensorflow.keras.models import Sequential, Model, load_model

data_path='c:/Users/eagle/Downloads/bitcamp/AI/_data/bitmart/'
save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

print("Keras: ", keras.__version__, "Tensorflow: ", tf.__version__)

# 데이터 로드
product_data = pd.read_csv(data_path + 'product.csv', sep=';', usecols=['seq', 'price', 'sale', 'category'])
views_data = pd.read_csv(data_path + 'views2.csv', sep=';', usecols=['user', 'product', 'views'])

# 사용자 및 제품 ID 인코딩
user_ids = views_data['user'].unique()
product_ids = np.union1d(views_data['product'].unique(), product_data['seq'].unique())
user_id_map = {id: i for i, id in enumerate(user_ids)}
product_id_map = {id: i for i, id in enumerate(product_ids)}

# 데이터셋 변환
views_data['user'] = views_data['user'].map(user_id_map)
views_data['product'] = views_data['product'].map(product_id_map)

# 모델 입력 데이터 생성
user_input = tf.keras.Input(shape=(), name='user')
product_input = tf.keras.Input(shape=(), name='product')

# 사용자 임베딩 레이어
user_embedding = Embedding(len(user_ids), 16)(user_input)
user_flatten = Flatten()(user_embedding)

# 제품 임베딩 레이어
product_embedding = Embedding(len(product_ids), 16)(product_input)
product_flatten = Flatten()(product_embedding)

# 사용자-제품 점수 예측
dot_product = Dot(axes=1)([user_flatten, product_flatten])
output = Dense(1, activation='linear')(dot_product)

# 모델 구성
#model=load_model(save_path+'bitmart_recommend2.h5')
model = Model(inputs=[user_input, product_input], outputs=output)
model.compile(loss='mse', optimizer='adam')

# 훈련 데이터 생성
X = [views_data['user'], views_data['product']]
y = views_data['views']
model.fit(X, y, epochs=1000)

# SavedModel로 저장
# tf.saved_model.save(model, save_path + 'bitmart_recommend')

#model.save(save_path+'bitmart_recommend2.h5')

user_id = 1  # 임의의 사용자 ID
user_input = np.array([user_id_map[user_id]])

# 모든 제품의 ID 배열 생성
all_product_ids = np.arange(len(product_ids))

# # 예측을 위한 제품 선택
# num_recommendations = 5  # 추천할 제품 수
# product_input = all_product_ids[:num_recommendations]  # 처음 5개의 제품 선택
# predictions = model.predict([user_input, product_input])

# 예측을 위한 제품 선택
num_recommendations = 5  # 추천할 제품 수
product_input = np.repeat(user_input, num_recommendations)
expanded_product_ids = np.tile(all_product_ids, len(user_input))
expanded_user_ids = np.repeat(user_input, len(all_product_ids))

predictions = model.predict([expanded_user_ids, expanded_product_ids])

# 예측 결과 확인
predictions = predictions.reshape((-1,))
top_recommendations = predictions.argsort()[-num_recommendations:][::-1]

print(f"사용자 ID: {user_id}")
print("최고 추천 제품:")
for product_idx in top_recommendations:
    product_id = product_ids[product_idx]
    print(f"- 제품 ID: {product_id}")