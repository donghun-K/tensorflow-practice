import tensorflow as tf

# model 만들기
model = tf.keras.models.Sequential([
  # 레이어 생성
  tf.keras.layers.Dense(64, activation='tanh'), # 첫 번째 파라미터는 레이어의 노드 수
  tf.keras.layers.Dense(128, activation='tanh'),
  tf.keras.layers.Dense(1, activation='sigmoid') # sigmoid : 결과 값을 0~1 사이에서 내주는 activation function
])

# model compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# adam: 가장 많이 쓰이는 optimizer
# binary_crossentropy: 결과가 0~1 사이인 분류 / 확률 문제에서 많이 쓰이는 loss function


# model 학습
model.fit(train_x, train_y, epochs=10) # epochs: 학습 횟수