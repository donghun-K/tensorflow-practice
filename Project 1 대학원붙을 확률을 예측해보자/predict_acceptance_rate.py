import tensorflow as tf
import pandas as pd

data = pd.read_csv("gpascore.csv")

data = data.dropna()  # 값이 없는 row나 column을 제거해주는 함수

# data = data.dropna(): 빈 칸 채우는 함수
# data['gpa']: 'gpa' column만 출력
# data['gpa'].min(): 'gpa' column의 최솟값
# data['gpa'].max(): 'gpa' column의 최댓값
# data['gpa'].count(): 'gpa' column의 개수


train_x = []
for i, rows in data.iterrows():
    train_x.append([rows["gre"], rows["gpa"], rows["rank"]])

train_y = data["admit"].values


# model 만들기
model = tf.keras.models.Sequential(
    [
        # 레이어 생성
        tf.keras.layers.Dense(64, activation="tanh"),  # 첫 번째 파라미터는 레이어의 노드 수
        tf.keras.layers.Dense(128, activation="tanh"),
        tf.keras.layers.Dense(
            1, activation="sigmoid"  # sigmoid : 결과 값을 0~1 사이에서 내주는 activation function
        ),
    ]
)

# model compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# adam: 가장 많이 쓰이는 optimizer
# binary_crossentropy: 결과가 0~1 사이인 분류 / 확률 문제에서 많이 쓰이는 loss function


# model 학습
model.fit(train_x, train_y, epochs=10)  # epochs: 학습 횟수
