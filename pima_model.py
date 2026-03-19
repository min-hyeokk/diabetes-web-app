# pima_model.py: 당뇨병 예측 모델 생성 및 저장

import sys
assert sys.version_info >= (3, 5) # Python 3.5 이상 필요

import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn 0.20 이상 필요

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0" # TensorFlow 2.0 이상 필요

import numpy as np
import os
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 결과 재현을 위한 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

# 그래프 기본 설정
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 1. 데이터 로드 (현재 폴더에 diabetes.csv가 있어야 함)
data = pd.read_csv('./diabetes.csv', sep=',')

print("\ndata.head(): \n", data.head())
data.describe()
data.info()

print("\n\nStep 2 - Prepare the data for the model building")
# X(8개 입력 특징)와 y(당뇨 여부 결과) 추출
X = data.values[:, 0:8]
y = data.values[:, 8]

# MinMaxScaler를 이용한 데이터 정규화(0~1 사이값)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# 학습용(67%)과 테스트용(33%) 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("\n\nStep 3 - Create and train the model")
# 딥러닝 모델 설계 (입력 8 -> 은닉1 12 -> 은닉2 8 -> 출력 1)
inputs = keras.Input(shape=(8,))
hidden1 = Dense(12, activation='relu')(inputs)
hidden2 = Dense(8, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = keras.Model(inputs, output)

model.summary()

# 모델 설정 및 학습 시작
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

# 학습 과정 시각화 (Loss & Accuracy 그래프)
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss', color=color)
ax1.plot(history.history['loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.plot(history.history['accuracy'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show() 

# 샘플 데이터 예측 테스트
X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n", np.round(model.predict(X_new), 2))

print("\nExporting SavedModels: ")
# 6-2 과제 핵심: 모델을 .keras 파일로 저장
model.save('pima_model.keras')

# 저장된 모델이 잘 불러와지는지 확인
model = keras.models.load_model('pima_model.keras')
X_new = X_test[:3]
print("\nnp.round(model.predict(X_new), 2): \n", np.round(model.predict(X_new), 2))