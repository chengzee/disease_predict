import pandas as pd
import numpy as np
import csv
from keras.layers import Dense, Lambda, dot, Activation, concatenate
from keras.layers import Layer
import keras.backend as K

# Parameters
# -------------------------------------------------------------------------------------------------------------------
bed = [631, 742, 701, 759, 765, 698]
lookback_days = 3
datasInADay = 288
input_dim = 2
secondsInADay = 60*60*24 
train_ratio = 0.7
neurals = [64, 128, 256, 512]
# # 定義 attention 機制 (return_sequence=True)
# class attention(Layer):
#     def __init__(self,**kwargs):
#         super(attention,self).__init__(**kwargs)
#     def build(self,input_shape):
#         self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
#         self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
#         super(attention, self).build(input_shape)
#     def call(self,x):
#         et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
#         at=K.softmax(et)
#         at=K.expand_dims(at,axis=-1)
#         output=x*at
#         return K.sum(output,axis=1, keepdims=True)
#     def compute_output_shape(self,input_shape):
#         return (input_shape)
#     def get_config(self):
#         return super(attention,self).get_config()


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
# print(targetRecent_arr)
# -------------------------------------------------------------------------------------------------------------------
# # 生成資料集
# def generator_with_augmentation(inputdata, starttime, lookback, dead_recently, samp_list_1, samp_list_0, targ_list_1, targ_list_0): # 輸入資料 samp_list = []; 輸出結果 targ_list = []
#     for i in range(datasInADay):
#         rows = np.arange(i+starttime, i+starttime+lookback)
#         if np.count_nonzero(inputdata[rows, 4] == 0) <= 316:
#             if dead_recently == 1:
#                 samp_list_1.append(inputdata[rows, 1:4])
#                 targ_list_1.append(dead_recently)
#             if dead_recently == 0:
#                 samp_list_0.append(inputdata[rows, 1:4])
#                 targ_list_0.append(dead_recently)
#     return  samp_list_1, samp_list_0, targ_list_1, targ_list_0
# 生成資料集
def generator_with_no_augmentation(inputdata, starttime, lookback, dead_recently, samp_list_1, samp_list_0, targ_list_1, targ_list_0): # 輸入資料 samp_list = []; 輸出結果 targ_list = []
    rows = np.arange(starttime, starttime+lookback)
    # if np.count_nonzero(inputdata[rows, 4] == 0) <= 316:
    if dead_recently == 1:
        samp_list_1.append(inputdata[rows, 1:input_dim+1])
        targ_list_1.append(dead_recently)
    if dead_recently == 0:
        samp_list_0.append(inputdata[rows, 1:input_dim+1])
        targ_list_0.append(dead_recently)
    return  samp_list_1, samp_list_0, targ_list_1, targ_list_0

samples_1 = []
samples_0 = []
targets_1 = []
targets_0 = []

# 測試結果csv建立
with open("predict_with_attention.csv", 'a+') as predictcsv:
    writer = csv.writer(predictcsv)
    writer.writerow(["第n次", "test_acc", "True Positive", "True Negative", "False Positive", "False Negative", "Precision_1", "Recall_1", "Precision_0", "Recall_0"])

for n in range(len(targetRecent_arr)):          # 近期死亡統計數量
    for m in range(len(bed)):                   # 試驗植床總共六床
        if targetRecent_arr[n, 2] == bed[m]:
            paddeddata_arr = np.array(pd.read_csv("addfeature9{}.csv".format(m+1)))
            # print("BedPlant:{}".format(m+1))
            # ----------------------------------------------------------------------------------------------------------------------------------------
            # 平均值正規化 [-1, 1]
            data_min = np.min(paddeddata_arr[:, 1:4], axis=0)
            data_max = np.max(paddeddata_arr[:, 1:4], axis=0)
            data_mean = np.mean(paddeddata_arr[:, 1:4], axis=0)
            # print(data_min)
            # print(data_max)
            # print(data_mean)
            paddeddata_arr[:, 1:4] = (paddeddata_arr[:, 1:4]-data_mean)/(data_max-data_min)
            # ----------------------------------------------------------------------------------------------------------------------------------------
            where = np.searchsorted(paddeddata_arr[:, 0], targetRecent_arr[n, 0]-secondsInADay*lookback_days) # 604800 是七天的秒數; 432000 是五天的秒數; 259200 是三天的秒數
            # print("where:{}".format(where))
            samples_1, samples_0, targets_1, targets_0 = generator_with_no_augmentation(paddeddata_arr, starttime=where, lookback=datasInADay*lookback_days, dead_recently=targetRecent_arr[n, 1], samp_list_1=samples_1, samp_list_0=samples_0, targ_list_1=targets_1, targ_list_0=targets_0)
# 轉為 numpy array
samples_1_arr = np.array(samples_1)
samples_0_arr = np.array(samples_0)
targets_1_arr = np.array(targets_1)
targets_0_arr = np.array(targets_0)
print("samples_1_arr.shape:{}".format(samples_1_arr.shape))
print("samples_0_arr.shape:{}".format(samples_0_arr.shape))
print("targets_1_arr.shape:{}".format(targets_1_arr.shape))
print("targets_0_arr.shape:{}".format(targets_0_arr.shape))

# print(np.count_nonzero(targets_1_arr==1))
# print(np.count_nonzero(targets_0_arr==0))
label1 = np.count_nonzero(targets_1_arr==1)
label0 = np.count_nonzero(targets_0_arr==0)

if label1>=label0:
    ratio = int(len(samples_1_arr)*train_ratio)
    # # # train test split
    x_train_arr = np.concatenate((samples_1_arr[:ratio], samples_0_arr[:ratio]), axis=0)
    y_train_arr = np.concatenate((targets_1_arr[:ratio], targets_0_arr[:ratio]), axis=0)
    x_test_arr = np.concatenate((samples_1_arr[-len(samples_0_arr[ratio:]):], samples_0_arr[ratio:]), axis=0)
    y_test_arr = np.concatenate((targets_1_arr[-len(samples_0_arr[ratio:]):], targets_0_arr[ratio:]), axis=0)
if label1<label0:
    ratio = int(len(samples_0_arr)*train_ratio)
    # # # train test split
    x_train_arr = np.concatenate((samples_1_arr[:ratio], samples_0_arr[:ratio]), axis=0)
    y_train_arr = np.concatenate((targets_1_arr[:ratio], targets_0_arr[:ratio]), axis=0)
    x_test_arr = np.concatenate((samples_1_arr[ratio:], samples_0_arr[-len(samples_1_arr[ratio:]):]), axis=0)
    y_test_arr = np.concatenate((targets_1_arr[ratio:], targets_0_arr[-len(samples_1_arr[ratio:]):]), axis=0)

# from sklearn.model_selection import train_test_split
# x_train_arr, x_val_arr, y_train_arr, y_val_arr = train_test_split(x_train_arr, y_train_arr, test_size=0.25)

from sklearn.utils import shuffle
x_train_arr, y_train_arr = shuffle(x_train_arr, y_train_arr, random_state=0)

print("x_train_arr.shape:{}".format(x_train_arr.shape))
print("y_train_arr.shape:{}".format(y_train_arr.shape))
print("x_test_arr.shape:{}".format(x_test_arr.shape))
print("y_test_arr.shape:{}".format(y_test_arr.shape))

# -------------------------------------------------------------------------------------------------------------------
# tf.keras model
for neural in neurals:
    for t in range(10):  # 做幾遍
        # LSTM 模型的訓練與驗證
        from keras.models import Sequential
        from keras import layers
        from keras.optimizers import RMSprop, Adam
        from keras.callbacks import ModelCheckpoint
        model = Sequential()
        model.add(layers.LSTM(neural,
                            input_shape=(datasInADay*lookback_days, input_dim), # (288*3, 3)
                            return_sequences=True,
                            dropout=0.2,
                            recurrent_dropout=0.2
                            ))
        model.add(layers.LSTM(neural,
                            return_sequences=True,
                            dropout=0.2,
                            recurrent_dropout=0.2
                            ))
        model.add(layers.LSTM(neural,
                            return_sequences=True,
                            dropout=0.2,
                            recurrent_dropout=0.2
                            ))
        model.add(layers.LSTM(neural,
                            return_sequences=True,
                            dropout=0.2,
                            recurrent_dropout=0.2
                            ))
        model.add(layers.LSTM(neural,
                            return_sequences=True,
                            dropout=0.2,
                            recurrent_dropout=0.2
                            ))
        model.add(layers.LSTM(neural,
                            return_sequences=True,
                            dropout=0.2,
                            recurrent_dropout=0.2
                            ))
        model.add(layers.LSTM(neural,
                            return_sequences=True,
                            dropout=0.2,
                            recurrent_dropout=0.2
                            ))
        model.add(layers.LSTM(neural,
                            return_sequences=True,
                            dropout=0.2,
                            recurrent_dropout=0.2
                            ))
        model.add(attention())
        # model.add(layers.LSTM(neural,
        #                     return_sequences=True,
        #                     # dropout=0.2
        #                     ))
        # model.add(layers.LSTM(neural,
        #                     return_sequences=False,
        #                     # dropout=0.2
        #                     ))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(optimizer=Adam(),
                    loss = 'binary_crossentropy',
                    metrics=['accuracy'])
    # -------------------------------------------------------------------------------------------------------------------
        # checkpoint
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, 
                                    monitor='val_accuracy', 
                                    verbose=1, 
                                    save_best_only=True,
                                    mode='max')
        callbacks_list = [checkpoint]
        # fit the model
        history = model.fit(x_train_arr, y_train_arr,
                            epochs=200,
                            batch_size=512,
                            # validation_data=(x_val_arr, y_val_arr),
                            validation_split=0.25, 
                            callbacks=callbacks_list,
                            verbose=1)
        model.load_weights("weights.best.hdf5")
        print("第{}次結果，選用最好的val_acc來對testSet做預測:".format(t+1))
        test_score = model.evaluate(x_test_arr, y_test_arr)
        print("test_score:{}".format(test_score))
        # 預測結果
        pred = model.predict(x_test_arr)
        TrueP = 0
        TrueN = 0
        FalseP = 0
        FalseN = 0 
        for pp in range(len(pred)):
            if(pred[pp]>0.5 and y_test_arr[pp]==1):
                TrueP += 1
            if(pred[pp]>0.5 and y_test_arr[pp]==0):
                FalseP += 1
            if(pred[pp]<=0.5 and y_test_arr[pp]==1):
                FalseN += 1
            if(pred[pp]<=0.5 and y_test_arr[pp]==0):
                TrueN += 1
        print("test數量:{}".format(len(x_test_arr)))
        print("True_Positive:{}".format(TrueP))
        print("True_Nagitive:{}".format(TrueN))
        print("False_Positive:{}".format(FalseP))
        print("False_Nagitive:{}".format(FalseN))
        # precision_1
        if TrueP+FalseP>0:
            precision_1 = TrueP/(TrueP+FalseP)
        else:
            precision_1 = 'nan'
        # recall_1
        if TrueP+FalseN>0:
            recall_1 = TrueP/(TrueP+FalseN)
        else:
            recall_1 = 'nan'
        # precision_0
        if TrueN+FalseN>0:
            precision_0 = TrueN/(TrueN+FalseN)
        else:
            precision_0 = 'nan'
        # recall_0
        if TrueN+FalseP>0:
            recall_0 = TrueN/(TrueN+FalseP)
        else:
            recall_0 = 'nan'
        print("Precision_1:{}".format(precision_1))
        print("Recall_1:{}".format(recall_1))
        print("Precision_0:{}".format(precision_0))
        print("Recall_0:{}".format(recall_0))
        with open("predict_with_attention.csv", 'a+') as predictcsv:
            writer = csv.writer(predictcsv)
            # writer.writerow(["第n次", "test_acc", "True Positive", "True Negative", "False Positive", "False Negative", "Precision_1", "Recall_1", "Precision_0", "Recall_0"])
            writer.writerow(["{},{}".format(t+1, neural), test_score[1], TrueP, TrueN, FalseP, FalseN, precision_1, recall_1, precision_0, recall_0])