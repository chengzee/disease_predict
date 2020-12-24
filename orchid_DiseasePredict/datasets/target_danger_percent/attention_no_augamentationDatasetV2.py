import pandas as pd
import numpy as np
import csv
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def __call__(self, hidden_states):
        """
        Many-to-one attention mechanism for Keras.
        @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
        @return: 2D tensor with shape (batch_size, 128)
        @author: felixhao28.
        """
        hidden_size = int(hidden_states.shape[2])
        # Inside dense layer
        #              hidden_states            dot               W            =>           score_first_part 
        # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
        # W is the trainable weight matrix of attention Luong's multiplicative style score
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        #            score_first_part           dot        last_hidden_state     => attention_weights
        # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        # attention_vec_reshape = Reshape((_delay, 1), name='attention_vec_reshape')(attention_vector)
        # return attention_vec_reshape
        return attention_vector

# Parameters
# -------------------------------------------------------------------------------------------------------------------
bed = [631, 742, 701, 759, 765, 698]
lookback_days = 3
datasInADay = 288
input_dim = 4
secondsInADay = 60*60*24 
train_ratio = 0.7
neurons = [64, 128, 256, 512]
test_times = 10
_epochs = 100
A_layers = 3

# 定義 attention 機制 (return_sequence=False)
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)
    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)
    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])
    def get_config(self):
        return super(attention,self).get_config()

# np.random.seed(1)
# 讀取 「(統計近期三日)近期死亡csv」
targetRecent = pd.read_csv("targetRecent.csv")
# 轉為 numpy array
targetRecent_arr = np.array(targetRecent)
# 對死亡率進行min-max normalization
# targetRecent_mean = np.mean(targetRecent_arr[:, 1], axis=0)
targetRecent_min = np.min(targetRecent_arr[:, 1], axis=0)
targetRecent_max = np.max(targetRecent_arr[:, 1], axis=0)
print("targetRecent_min:{}".format(targetRecent_min))
print("targetRecent_max:{}".format(targetRecent_max))
targetRecent_arr[:, 1] = (targetRecent_arr[:, 1]-targetRecent_min)/(targetRecent_max-targetRecent_min)
# print(np.mean(targetRecent_arr[:, 1], axis=0))
# print(np.std(targetRecent_arr[:,1], axis=0))

# print(targetRecent_arr)
# -------------------------------------------------------------------------------------------------------------------
# 生成資料集
def generator_with_no_augmentation(inputdata, starttime, lookback, dead_recently_rate, samp_list, targ_list): # 輸入資料 samp_list = []; 輸出結果 targ_list = []
    rows = np.arange(starttime, starttime+lookback)
    # if np.count_nonzero(inputdata[rows, 4] == 0) <= 316:
    samp_list.append(inputdata[rows, 1:input_dim+1])
    targ_list.append(dead_recently_rate)
    return  samp_list, targ_list
# 生成資料集
# def generator_with_augmentation(inputdata, starttime, lookback, dead_recently, samp_list_1, samp_list_0, targ_list_1, targ_list_0): # 輸入資料 samp_list = []; 輸出結果 targ_list = []
#     for i in range(datasInADay):
#         rows = np.arange(i+starttime, i+starttime+lookback)
#         if np.count_nonzero(inputdata[rows, 4] == 0) <= 316:
#             if dead_recently == 1:
#                 samp_list_1.append(inputdata[rows, 1:input_dim+1])
#                 targ_list_1.append(dead_recently)
#             if dead_recently == 0:
#                 samp_list_0.append(inputdata[rows, 1:input_dim+1])
#                 targ_list_0.append(dead_recently)
#     return  samp_list_1, samp_list_0, targ_list_1, targ_list_0
# # 生成資料集
# def generator_with_no_augmentation(inputdata, starttime, lookback, dead_recently, samp_list_1, samp_list_0, targ_list_1, targ_list_0): # 輸入資料 samp_list = []; 輸出結果 targ_list = []
#     rows = np.arange(starttime, starttime+lookback)
#     # if np.count_nonzero(inputdata[rows, 4] == 0) <= 316:
#     if dead_recently == 1:
#         samp_list_1.append(inputdata[rows, 1:input_dim+1])
#         targ_list_1.append(dead_recently)
#     if dead_recently == 0:
#         samp_list_0.append(inputdata[rows, 1:input_dim+1])
#         targ_list_0.append(dead_recently)
#     return  samp_list_1, samp_list_0, targ_list_1, targ_list_0

# samples_1 = []
# samples_0 = []
# targets_1 = []
# targets_0 = []
samples = []
targets = []
# 測試結果csv建立
# with open("{}neurons_{}LSTM_noDropout_1Att_2Dense.csv".format(neuron, A+1), 'a+') as predictcsv:
#     writer = csv.writer(predictcsv)
#     writer.writerow(["第n次", "test_loss", "test_mae"])
#     # writer.writerow(["第n次", "test_loss", "test_mae"])

for n in range(len(targetRecent_arr)):          # 近期死亡統計數量
    for m in range(len(bed)):                   # 試驗植床總共六床
        if targetRecent_arr[n, 2] == bed[m]:
            paddeddata_arr = np.array(pd.read_csv("addfeature9{}.csv".format(m+1))) # addfeature9*_arr.shape = (datas, 5)
            # print(paddeddata_arr.shape)
            # print("BedPlant:{}".format(m+1))
            source_dimension = len(paddeddata_arr[0, :])
            # print(source_dimension)
            # ----------------------------------------------------------------------------------------------------------------------------------------
            # 平均值正規化 [-1, 1]
            data_min = np.min(paddeddata_arr[:, 1:source_dimension], axis=0)
            data_max = np.max(paddeddata_arr[:, 1:source_dimension], axis=0)
            data_mean = np.mean(paddeddata_arr[:, 1:source_dimension], axis=0)
            print(data_min)
            print(data_max)
            # print(data_mean)
            paddeddata_arr[:, 1:source_dimension] = (paddeddata_arr[:, 1:source_dimension]-data_min)/(data_max-data_min)
            # ----------------------------------------------------------------------------------------------------------------------------------------
            where = np.searchsorted(paddeddata_arr[:, 0], targetRecent_arr[n, 0]-secondsInADay*lookback_days) # 604800 是七天的秒數; 432000 是五天的秒數; 259200 是三天的秒數
            # print("where:{}".format(where))
            samples, targets = generator_with_no_augmentation(paddeddata_arr, starttime=where, lookback=datasInADay*lookback_days, dead_recently_rate=targetRecent_arr[n, 1], samp_list=samples, targ_list=targets)
# 轉為 numpy array
samples_arr = np.array(samples)
targets_arr = np.array(targets)
print("samples_arr.shape:{}".format(samples_arr.shape))
print("targets_arr.shape:{}".format(targets_arr.shape))

from sklearn.model_selection import train_test_split
x_train_arr, x_test_arr, y_train_arr, y_test_arr = train_test_split(samples_arr, targets_arr, test_size=0.25)

# from sklearn.utils import shuffle
# x_train_arr, y_train_arr = shuffle(x_train_arr, y_train_arr, random_state=0)

print("x_train_arr.shape:{}".format(x_train_arr.shape))
print("y_train_arr.shape:{}".format(y_train_arr.shape))
print("x_test_arr.shape:{}".format(x_test_arr.shape))
print("y_test_arr.shape:{}".format(y_test_arr.shape))

import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------
# tf.keras model

for A in range(3, 4):
    for neuron in neurons:
        total_loss = np.zeros((_epochs))
        total_val_loss = np.zeros((_epochs))
        total_test_loss = 0
        total_test_mae = 0
        for t in range(test_times):  # 做幾遍
            # LSTM 模型的訓練與驗證
            from tensorflow.keras.models import Sequential
            from keras import layers
            from tensorflow.keras.optimizers import RMSprop, Adam
            from tensorflow.keras.callbacks import ModelCheckpoint
            model = Sequential()
            model.add(layers.LSTM(neuron,
                                input_shape=(datasInADay*lookback_days, input_dim), # (288*3, 3)
                                return_sequences=True,
                                ))
            for aa in range(A):
                model.add((layers.LSTM(neuron, 
                                return_sequences=True,
                                dropout=0.2,
                                recurrent_dropout=0.2
                                )))
            model.add(Attention())
            # model.add(layers.Dense(32, activation='relu'))
            # model.add(layers.Dense(1, activation='sigmoid'))
            # 迴歸問題最後輸出為線性輸出，最後一層的輸出層不會有啟動函數，此為純量回歸的基本設定。
            model.add(layers.Dense(1))
            model.summary()
            model.compile(optimizer=Adam(),
                        loss = 'mse',
                        # metrics=['mae']
                        )
        # -------------------------------------------------------------------------------------------------------------------

            # checkpoint
            filepath="weights.best.hdf5"
            checkpoint = ModelCheckpoint(filepath, 
                                        monitor='val_loss', 
                                        verbose=1, 
                                        save_best_only=True,
                                        mode='min')
            callbacks_list = [checkpoint]
            # fit the model
            history = model.fit(x_train_arr, y_train_arr,
                                epochs=_epochs,
                                batch_size=512,
                                # validation_data=(x_val_arr, y_val_arr),
                                validation_split=0.25, 
                                callbacks=callbacks_list,
                                verbose=1)
            model.load_weights("weights.best.hdf5")
            print("第{}次結果，選用最低的val_loss來對testSet做預測:".format(t+1))
            test_mse_results = model.evaluate(x_test_arr, y_test_arr)
            print("test_mse_results:{}".format(test_mse_results))
            # 預測結果
            pred = model.predict(x_test_arr)
            print("pred_results:{}".format(pred))
            total_test_loss += test_mse_results
            test_mae_results = (np.sum(np.abs(pred-y_test_arr)))/len(y_test_arr)
            total_test_mae += test_mae_results
            with open("{}neurons_{}LSTM_noDropout_1Att_2Dense.csv".format(neuron, A+1), 'a+') as predictcsv:
                writer = csv.writer(predictcsv)
                # writer.writerow(["第n次", "test_loss", "test_mae"])
                writer.writerow(["{},{}".format(t+1, neuron), test_mse_results, test_mae_results])
            total_loss += np.array(history.history["loss"])
            total_val_loss += np.array(history.history["val_loss"])
        mean_mse = total_test_loss/test_times
        mean_mae = total_test_mae/test_times
        with open("{}neurons_{}LSTM_noDropout_1Att_2Dense.csv".format(neuron, A+1), 'a+') as predictcsv:
            writer = csv.writer(predictcsv)
            # writer.writerow(["第n次", "test_loss", "test_mae"])
            writer.writerow(["mean,{}".format(neuron), mean_mse, mean_mae])
        epochs = range(1, len(total_loss)+1)
        mean_loss = total_loss/test_times
        mean_val_loss = total_val_loss/test_times
        plt.figure()
        plt.plot(epochs, mean_loss, 'bo', label="Training loss")
        plt.plot(epochs, mean_val_loss, 'ro', label="Validation loss")
        plt.title("Training and validation loss (test {} time)".format(test_times))
        plt.xlabel("epochs")
        plt.ylabel("Mean Square Error(MSE)")
        plt.legend()
        plt.savefig("{}neurons_{}layers_LSTM_noDropout_1Att_2Dense_mean_of_{}_times_loss.png".format(neuron, A+1, test_times))
