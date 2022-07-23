from pickletools import optimize
import tensorflow as tf
import matplotlib.pyplot as plt

(train_x, train_y,), (
    test_x,
    test_y,
) = tf.keras.datasets.fashion_mnist.load_data()  # 구글이 기본으로 제공해주는 dataset

# plt.imshow(train_x[0]): pixel data를 이미지로 보여줌
# plt.show()

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

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            128, input_shape=(28, 28), activation="relu"  # input_shape: input data의 모양
        ),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Flatten(),  # 다차원 행렬을  1차원으로 압축
        tf.keras.layers.Dense(
            len(class_names),  # 마지막 레이어의 노드 수는 카테고리의 개수,
            activation="softmax",  # softmax: 결과를 0~1로 압축, 카테고리 예측 문제에 사용하는 activation function
        ),
    ]
)

model.summary()  # model의 아웃라인 보여주기

exit()

model.compile(
    loss="sparse_categorical_crossentropy",  # 카테고리 예측 문제에 쓰이는 loss function
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(train_x, train_y, epochs=5)
