import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(train_x, train_y,), (
    test_x,
    test_y,
) = tf.keras.datasets.fashion_mnist.load_data()  # 구글이 기본으로 제공해주는 dataset

# plt.imshow(train_x[0]): pixel data를 이미지로 보여줌
# plt.show()

train_x = train_x / 255.0  # pixel data를 0~255에서 0~1로 압축
text_x = test_x / 255.0

train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))  # numpy array의 shape 변경
test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankleboot",
]

# Convolution model 만들기!
# Conv -> Pooling -> Flatten -> Dense -> 출력!
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(  # convolution layer
            32, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)
        ),  # 이미지 데이터는 음의 값이 없기 때문에 음의 값이 나오지 않도록 relu 사용
        # tf.keras.layers.Dense(128, input_shape=(28, 28), activation="relu"), input_shape: input data의 모양
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),  # 다차원 행렬을  1차원으로 압축
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(
            len(class_names),  # 마지막 레이어의 노드 수는 카테고리의 개수,
            activation="softmax",  # softmax: 결과를 0~1로 압축, 카테고리 예측 문제에 사용하는 activation function
        ),
    ]
)

model.summary()  # model의 아웃라인 보여주기

model.compile(
    loss="sparse_categorical_crossentropy",  # 카테고리 예측 문제에 쓰이는 loss function
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(
    train_x, train_y, validation_data=(test_x, test_y), epochs=5
)  # validation_data: epoch 1회 끝날 때마다 모델 평가

# score = model.evaluate(test_x, test_y) 모델 정확도 평가
# print(score)
# training accuracy > test accuracy가 되는 현상을 overfitting이라고 함


model.save("model1")  # 전체 모델 저장하기
