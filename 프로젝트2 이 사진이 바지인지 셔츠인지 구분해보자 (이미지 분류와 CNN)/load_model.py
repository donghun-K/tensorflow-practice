import tensorflow as tf


model = tf.keras.models.load_model("model1")  # 모델 불러오기

model.summary()
