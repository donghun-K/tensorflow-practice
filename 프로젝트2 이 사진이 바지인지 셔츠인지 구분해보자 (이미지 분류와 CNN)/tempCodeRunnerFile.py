model2 = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(
            len(class_names),
            activation="softmax",
        ),
    ]
)
model2.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
model2.load_weights("checkpoint/weights1")
model2.evaluate(test_x, test_y)