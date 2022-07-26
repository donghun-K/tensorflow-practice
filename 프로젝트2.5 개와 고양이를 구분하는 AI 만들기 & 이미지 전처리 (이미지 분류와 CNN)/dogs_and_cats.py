# import os
# import shutil

# for i in os.listdir("dataset/train"):
#     if "cat" in i:
#         shutil.move("dataset/train/" + i, "./dataset/cat/" + i)
#     if "dog" in i:
#         shutil.move("dataset/train/" + i, "./dataset/dog/" + i)

import tensorflow as tf

train_ds = (
    tf.keras.preprocessing.image_dataset_from_directory(  # 폴더 내 이미지를 Dataset으로 만들어줌
        "dataset",
        image_size=(64, 64),
        batch_size=32,
        subset="training",  # validation dataset 나누기
        validation_split=0.2,  # 전체 데이터의 20%
        seed=1234,
    )
)

val_ds = (
    tf.keras.preprocessing.image_dataset_from_directory(  # 폴더 내 이미지를 Dataset으로 만들어줌
        "dataset",
        image_size=(64, 64),
        batch_size=32,
        subset="validation",
        validation_split=0.2,
        seed=1234,
    )
)
