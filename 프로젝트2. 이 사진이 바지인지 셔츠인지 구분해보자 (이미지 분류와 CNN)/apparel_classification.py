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
