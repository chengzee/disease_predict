import pandas as pd
import numpy as np
# There are so many batches (ex. HT7B01, HT7C01, HT8101, HT8201, 
# HT8301, HT8401, HT8501, HT8601, HT8701, HT8901, HT8A01, HT8B01)
# at least 1 to 12 data per batch
# only give seven days per data

# 讀病害資料
df1 = pd.read_excel("2019BH_delet_repeatday.xlsx") # 死亡數量&日期
df2 = pd.read_csv("2019Indoor_env.csv") # 氣候條件
array1 = np.array(df1)
array2 = np.array(df2)
X = []
Y = []

for n in range(361):
    X.append(array2[n*288:(n+5)*288, 1:].reshape((-1, 3)))
    same_flag = 0
    for m in range(len(array1[:, 0])):
        if array2[n*288, 0][3:5]+array2[n*288, 0][2]+array2[n*288, 0][0:2] == str(array1[m, 0])[5:10]:
            same_flag = 1
            # Y.append(array1[m, 8])
            Y.append(np.array((1, 0)))
    if same_flag == 0:
        Y.append(np.array((0, 1)))
# PS 刪去了 6 筆在同一天有多筆的資料，僅留其中一筆
# become array
X = np.array(X, dtype=np.float64)
Y = np.array(Y, dtype=np.float64)
print("X_array's shape:{}".format(X.shape)) # (361, 1440, 3)
print("Y_array's shape:{}".format(Y.shape)) # (361, )
print(X)
print(Y)
# =====================================================================================
# Normalization
'''
降維前作正規化
'''
# 使用 sklearn API 作正規化 
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# Z_sk = scaler.fit_transform(X)
# print(Z_sk)
# print(Z_sk.shape)
# # 將 nan 數值轉為 0
# Z_sk[np.isnan(Z_sk)==True] = 0
# =====================================================================================
'''
透過 scikit-learn 將 4320 維的氣候資訊降到 2 維
'''
# from sklearn.decomposition import PCA
# # 只取最大的兩個主成分， scikit-learn 會自動排序出最大的前兩個 eigenvalue 及其對應的 eigenvector
# n_components = 2
# random_state = 9527
# pca = PCA(n_components=n_components, random_state=random_state)
# # 對正規化後的特徵 Z 做 PCA
# L = pca.fit_transform(Z_sk) # (n_sample, n_components)

# import matplotlib.pyplot as plt
# # 將投影到第一主成分的 repr. 顯示在 x 軸，第二主成分在 y 軸
# plt.figure()
# plt.scatter(L[:, 0], L[:, 1], c=Y)
# plt.axis('equal') 
# # plt.legend()
# plt.show()
# # 前 15 主成分分析
# pca_15d = PCA(15, random_state=random_state)
# pca_15d.fit(Z_sk)
# print(np.round(pca_15d.explained_variance_ratio_, 3))
# =====================================================================================
# split train and validation
from sklearn.model_selection import train_test_split
x_train, x_validation, y_train, y_validation =  train_test_split(X, Y, test_size=0.2, random_state=3)

# import packages
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization, concatenate, AveragePooling2D, Embedding
from keras.layers import LSTM, RNN, LSTMCell
from keras.optimizers import RMSprop

# =====================================================================================
# Training parameters
# learning_rate = 0.001
# training_steps = 10000
batch_size = 32 # 128
# display_step = 200

# Network parameters
num_input = 3 # data input (data input's shape = (1440*3))
timesteps = 1440 # timesteps
num_hidden_units = 256 # hidden layer num of feature
num_class = 2 # labels are 2 kind ([1, 0]-->disease or [0, 1]-->health)

# build the RNN model
# def simpleLSTM(allow_cudnn_kernel=True):
#     # CuDNN is only available at the layer level, and not at the cell level.
#     # This means `LSTM(units)` will use the CuDNN kernel,
#     # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
#     if allow_cudnn_kernel:
#         # The LSTM layer with default options uses CuDNN.
#         print("CuDNN used!")
#         lstm_layer = LSTM(num_hidden_units, input_shape = (timesteps, num_input)) # [batch, timesteps, feature]
#     else:
#         # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
#         lstm_layer = RNN(LSTMCell(num_hidden_units, input_shape = (timesteps, num_input))) # [batch, timesteps, feature]
#     simpleLSTM = Sequential()
#     simpleLSTM.add(lstm_layer)
#     # simpleLSTM.add(BatchNormalization())
#     # simpleLSTM.add(Dense(num_class, activation='sigmoid'))

#     simpleLSTM.add(Dense(num_class, activation='sigmoid'))
#     simpleLSTM.summary()
#     return simpleLSTM

def simpleLSTM(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    # if allow_cudnn_kernel:
    #     # The LSTM layer with default options uses CuDNN.
    #     print("CuDNN used!")
    #     lstm_layer = LSTM(num_hidden_units, input_shape = (timesteps, num_input)) # [batch, timesteps, feature]
    # else:
    #     # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
    #     lstm_layer = RNN(LSTMCell(num_hidden_units, input_shape = (timesteps, num_input))) # [batch, timesteps, feature]
    simpleLSTM = Sequential()
    simpleLSTM.add(LSTM(num_hidden_units, input_shape = (timesteps, num_input), return_sequences=True)) # [batch, timesteps, feature]
    # model.add(Dropout(0.2))
    # simpleLSTM.add(BatchNormalization())
    # simpleLSTM.add(Dense(num_class, activation='sigmoid'))
    simpleLSTM.add(LSTM(128, return_sequences=True)) # [batch, timesteps, feature]
    simpleLSTM.add(LSTM(128)) # [batch, timesteps, feature]
    # simpleLSTM.add(Dropout(0.2))
    simpleLSTM.add(Dense(num_class, activation='sigmoid'))
    simpleLSTM.summary()
    return simpleLSTM

model = simpleLSTM()
# model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
#                 optimizer='sgd',
#                 metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
model.fit(x_train, y_train, 
          validation_data=(x_validation, y_validation), 
          epochs=50, 
          batch_size=batch_size)
# =====================================================================================
# input_dim = (1440, 3)
# output_size = 1 # Labels are between 0 and 1
# # build the basic machine learning model
# def basic_ML():
#     basic_ML = Sequential()
#     # basic_ML.add(Embedding(1440, 64))
#     basic_ML.add(Flatten(input_shape=input_dim))
#     basic_ML.add(Dense(32, activation='relu'))
#     basic_ML.add(Dense(output_size))
#     basic_ML.summary()
#     return basic_ML

# model = basic_ML()
# model.compile(loss='mae', 
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
# history = model.fit(x_train, y_train, 
#                     validation_data=(x_validation, y_validation), 
#                     epochs=20)
# =====================================================================================

# plot loss curve

import matplotlib.pyplot as plt

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

from mpl_toolkits.mplot3d import Axes3D

plt.figure()
ax = plt.subplot(111, projection='3d')
ax.scatter(X[0, :, 0], X[0, :, 1], X[0, :, 2], c='b')
plt.show()

