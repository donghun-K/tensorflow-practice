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


# 전처리
def pre_processing(img, y):
    img = tf.cast(img / 255.0, tf.float32)
    return (img, y)


train_ds = train_ds.map(pre_processing)
val_ds = train_ds.map(pre_processing)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu", input_shape=(64, 64, 3)
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", activation="relu", input_shape=(64, 64, 3)
        ),
        tf.keras.layers.Dropout(0.2),  # 레이어의 20%를 제거, Over fitting 방지
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=5)
