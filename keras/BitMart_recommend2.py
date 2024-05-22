import numpy as np
import tensorflow as tf
import pandas as pd

data_path='c:/Users/eagle/Downloads/bitcamp/AI/_data/bitmart/'
save_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/' 
save_mcp_path='c:/Users/eagle/Downloads/bitcamp/AI/_save/MCP/'

# 데이터 불러오기
product_data = pd.read_csv(data_path + 'product.csv', sep=';', usecols=['seq', 'price', 'sale', 'category'])
views_data = pd.read_csv(data_path + 'views.csv', sep=';', usecols=['user', 'product', 'views'])

# 고유한 유저와 상품 ID 추출
unique_user_ids = np.unique(views_data['user'])
unique_product_ids = np.unique(views_data['product'])

# 유저 및 상품 ID를 정수형으로 매핑
user_id_map = {id: i for i, id in enumerate(unique_user_ids)}
product_id_map = {id: i for i, id in enumerate(unique_product_ids)}

# 매핑된 ID로 데이터 재인코딩
views_data['user_id'] = views_data['user'].map(user_id_map)
views_data['product_id'] = views_data['product'].map(product_id_map)

# 모델 입력 데이터 생성
user_input = tf.keras.Input(shape=(), dtype=tf.int32, name='user_id')
product_input = tf.keras.Input(shape=(), dtype=tf.int32, name='product_id')

# 유저 임베딩 레이어
user_embedding = tf.keras.layers.Embedding(len(unique_user_ids), embedding_dim)(user_input)
user_flattened = tf.keras.layers.Flatten()(user_embedding)

# 상품 임베딩 레이어
product_embedding = tf.keras.layers.Embedding(len(unique_product_ids), embedding_dim)(product_input)
product_flattened = tf.keras.layers.Flatten()(product_embedding)

# 유저와 상품의 임베딩 특성을 결합
concatenated = tf.keras.layers.Concatenate()([user_flattened, product_flattened])

# MLP 레이어
dense1 = tf.keras.layers.Dense(128, activation='relu')(concatenated)
dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
output = tf.keras.layers.Dense(1)(dense2)

# 모델 생성
model = tf.keras.Model(inputs=[user_input, product_input], outputs=output)

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 데이터셋 생성
dataset = tf.data.Dataset.from_tensor_slices((views_data['user_id'].values, views_data['product_id'].values, views_data['views'].values))
dataset = dataset.shuffle(len(views_data)).batch(batch_size)

# 모델 훈련
model.fit(dataset, epochs=num_epochs)

# 임의의 유저에 대한 추천 상품 예측
user_id = 1  # 임의의 유저 ID
user_input = np.array([user_id_map[user_id]])  # 유저 ID를 매핑된 정수형으로 변환
product_ids = np.arange(len(unique_product_ids))  # 모든 상품 ID
product_inputs = np.tile(user_input, len(unique_product_ids))  # 유저 ID를 상품 수만큼 반복하여 배열 생성

predictions = model.predict([product_inputs, product_ids])  # 추천 상품 평점 예측
recommended_product_id = unique_product_ids[np.argmax(predictions)]  # 가장 높은 평점을 가진 상품 ID 선택

# 추천 상품 정보 출력
recommended_product = product_data[product_data['seq'] == recommended_product_id]
print(recommended_product)