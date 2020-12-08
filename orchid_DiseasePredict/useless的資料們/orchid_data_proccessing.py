import pandas as pd
import numpy as np
# There are so many batches (ex. HT7B01, HT7C01, HT8101, HT8201, 
# HT8301, HT8401, HT8501, HT8601, HT8701, HT8901, HT8A01, HT8B01)
# at least 1 to 12 data per batch
# only give seven days per data

# 讀病害資料
df1 = pd.read_excel("2019BH.xlsx") # 死亡數量&日期
df2 = pd.read_csv("2019Indoor_env.csv") # 氣候條件
array1 = np.array(df1)
array2 = np.array(df2)
X = []
Y = []

for n in range(len(array1[:, 0])):
    count = 0
    # print(str(array1[n, 0])[5:10])
    for m in range(len(array2[:, 0])):
        if str(array1[n, 0])[5:10] == array2[m, 0][3:5]+array2[m, 0][2]+array2[m, 0][0:2]:
            count += 1
            print(array2[m, 0][3:5]+array2[m, 0][2]+array2[m, 0][0:2])
            if count == 288:
                X.append(array2[m-1440:m, 1:4])
                if array1[n, 8] > -4:
                    Y.append(0)
                else:
                    Y.append(1)

# become array
X = np.array(X)
Y = np.array(Y)
print("X_array's shape:{}".format(X.shape))
print("Y_array's shape:{}".format(Y.shape))
print(X)
# =====================================================================================
# Normalization
for n in range(len(X[:, 0, 0])):
    mu0 = np.mean(X[n, :, 0])
    std0 = np.std(X[n, :, 0])
    X[n, :, 0] = (X[n, :, 0]-mu0)/std0
    
    mu1 = np.mean(X[n, :, 1])
    std1 = np.std(X[n, :, 1])
    X[n, :, 1] = (X[n, :, 1]-mu1)/std1
    
    mu2 = np.mean(X[n, :, 2])
    std2 = np.std(X[n, :, 2])
    X[n, :, 2] = (X[n, :, 2]-mu2)/std2
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
# input_dim = (1440, 3)
# units = 128
# output_size = 2 # labels are from 0 and 1
# # build the RNN model
# def simpleLSTM(allow_cudnn_kernel=True):
#     # CuDNN is only available at the layer level, and not at the cell level.
#     # This means `LSTM(units)` will use the CuDNN kernel,
#     # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
#     if allow_cudnn_kernel:
#         # The LSTM layer with default options uses CuDNN.
#         print("CuDNN used!")
#         lstm_layer = LSTM(units, input_shape=input_dim)
#     else:
#         # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
#         lstm_layer = RNN(LSTMCell(units), input_shape=input_dim)
#     simpleLSTM = Sequential()
#     # simpleLSTM.add(Embedding(1440, 64))
#     simpleLSTM.add(lstm_layer)
#     simpleLSTM.add(BatchNormalization())
#     simpleLSTM.add(Dense(output_size))
#     simpleLSTM.summary()
#     return simpleLSTM

# model = simpleLSTM()
# model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
#                 optimizer='sgd',
#                 metrics=['accuracy'])
# model.fit(x_train, y_train, 
#           validation_data=(x_validation, y_validation), 
#           epochs=50)
# =====================================================================================
input_dim = (1440, 3)
output_size = 1 # Labels are between 0 and 1
# build the basic machine learning model
def basic_ML():
    basic_ML = Sequential()
    # basic_ML.add(Embedding(1440, 64))
    basic_ML.add(Flatten(input_shape=input_dim))
    basic_ML.add(Dense(32, activation='relu'))
    basic_ML.add(Dense(output_size))
    basic_ML.summary()
    return basic_ML

model = basic_ML()
model.compile(loss='mae', 
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, 
                    validation_data=(x_validation, y_validation), 
                    epochs=20)
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

