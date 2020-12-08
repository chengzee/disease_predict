import pandas as pd
import numpy as np
import csv

from keras.layers import Dense, Lambda, dot, Activation, concatenate
from keras.layers import Layer

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
        attention_vector = Dense(32, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

bed = [631, 742, 701, 759, 765, 698]
# np.random.seed(1)
targetRecent = pd.read_csv("targetRecent.csv")
targetRecent_arr = np.array(targetRecent)
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# -------------------------------------------------------------------------------------------------------------------
# train test split
for t in range(20):          #想做幾遍#
    from sklearn.model_selection import train_test_split
    y_train_tR_arr, y_test_tR_arr = train_test_split(targetRecent_arr, test_size=0.33, random_state=2)
    partial_y_train_tR_arr, y_val_tR_arr = train_test_split(y_train_tR_arr, test_size=0.33, random_state=t+30) # random一波
    # print(y_train)
    # print(y_test)
    def generator_with_augmentation(inputdata, starttime, lookback, dead_recently, samp_list, targ_list):
        # samples = [] # 輸入資料
        # targets = [] # 輸出結果
        for i in range(288):
            rows = np.arange(i+starttime, i+starttime+lookback)
            samp_list.append(inputdata[rows, 1:5])
            targ_list.append(dead_recently)
        return samp_list, targ_list
    train_samples = []
    train_targets = []
    for n in range(len(partial_y_train_tR_arr)):          # the recent dead
        for m in range(len(bed)):                   # total plant bed length (6)
            if partial_y_train_tR_arr[n, 2] == bed[m]:
                paddeddata_arr = np.array(pd.read_csv("addfeature9{}.csv".format(m+1)))
                # print("BedPlant:{}".format(m+1))
                where = np.searchsorted(paddeddata_arr[:, 0], partial_y_train_tR_arr[n, 0]-259200) # 604800 是七天的秒數; 432000 是五天的秒數; 259200 是三天的秒數
                # print("where:{}".format(where))
                train_samples, train_targets = generator_with_augmentation(paddeddata_arr, starttime=where, lookback=288*3, dead_recently=partial_y_train_tR_arr[n, 1], samp_list=train_samples, targ_list=train_targets)
    val_samples = []
    val_targets = []
    for n in range(len(y_val_tR_arr)):          # the recent dead
        for m in range(len(bed)):                   # total plant bed length (6)
            if y_val_tR_arr[n, 2] == bed[m]:
                paddeddata_arr = np.array(pd.read_csv("addfeature9{}.csv".format(m+1)))
                # print("BedPlant:{}".format(m+1))
                where = np.searchsorted(paddeddata_arr[:, 0], y_val_tR_arr[n, 0]-259200) # 604800 是七天的秒數; 432000 是五天的秒數; 259200 是三天的秒數
                # print("where:{}".format(where))
                val_samples, val_targets = generator_with_augmentation(paddeddata_arr, starttime=where, lookback=288*3, dead_recently=y_val_tR_arr[n, 1], samp_list=val_samples, targ_list=val_targets)
    test_samples = []
    test_targets = []
    for n in range(len(y_test_tR_arr)):          # the recent dead
        for m in range(len(bed)):                   # total plant bed length (6)
            if y_test_tR_arr[n, 2] == bed[m]:
                paddeddata_arr = np.array(pd.read_csv("addfeature9{}.csv".format(m+1)))
                # print("BedPlant:{}".format(m+1))
                where = np.searchsorted(paddeddata_arr[:, 0], y_test_tR_arr[n, 0]-259200) # 604800 是七天的秒數; 432000 是五天的秒數; 259200 是三天的秒數
                # print("where:{}".format(where))
                test_samples, test_targets = generator_with_augmentation(paddeddata_arr, starttime=where, lookback=288*3, dead_recently=y_test_tR_arr[n, 1], samp_list=test_samples, targ_list=test_targets)
    x_train = np.array(train_samples)
    y_train = np.array(train_targets)
    x_val = np.array(val_samples)
    y_val = np.array(val_targets)
    x_test = np.array(test_samples)
    y_test = np.array(test_targets)

    # print("x_train.shape:{}".format(x_train.shape))
    # print("y_train.shape:{}".format(y_train.shape))
    # print("x_val.shape:{}".format(x_val.shape))
    # print("y_val.shape:{}".format(y_val.shape))
    # print("x_test.shape:{}".format(x_test.shape))
    # print("y_test.shape:{}".format(y_test.shape))

    # Normalize
    mean = (np.concatenate((x_train, x_val), axis=0)).mean(axis=0)
    # print("x_train mean:{}".format(mean))
    std = (np.concatenate((x_train, x_val), axis=0)).std(axis=0)
    # print("std:{}".format(std))
    x_train = (x_train-mean)/std
    x_val = (x_val-mean)/std
    x_test = (x_test-mean)/std
    # ----------------------------------------------------------------------------------------------------------------------------------------
    # LSTM 模型的訓練與驗證
    from keras.models import Sequential
    from keras import layers
    from keras.optimizers import RMSprop, Adam
    from keras.callbacks import ModelCheckpoint

    model = Sequential()
    model.add(layers.LSTM(32,
                        input_shape=(288*3, 4), 
                        return_sequences=True
                        ))
    model.add(layers.LSTM(32,
                        return_sequences=True
                        ))
    model.add(Attention(name='attention_weight'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer=Adam(),
                loss = 'binary_crossentropy',
                metrics=['accuracy'])
    # # # ------------------------------------------------------------------------------------

    # checkpoint
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, 
                                monitor='val_accuracy', 
                                verbose=1, 
                                save_best_only=True,
                                mode='max')
    callbacks_list = [checkpoint]
    # fit the model
    history = model.fit(x_train, y_train,
                        epochs=200,
                        batch_size=512,
                        callbacks=callbacks_list,
                        validation_data=(x_val, y_val), 
                        verbose=1)

    model.load_weights("weights.best.hdf5")
    print("第{}次結果，選用最好的val_acc來對testSet做預測:".format(t+1))
    # scores1 = model.evaluate(x_train, y_train)
    # print("training_score:{}".format(scores1))
    # scores2 = model.evaluate(x_val, y_val)
    # print("val_score:{}".format(scores2))
    scores3 = model.evaluate(x_test, y_test)
    print("test_score:{}".format(scores3))
    
    pred = model.predict(x_test)
    TrueP = 0
    TrueN = 0
    FalseP = 0
    FalseN = 0 
    for pp in range(len(pred)):
        if(pred[pp]>0.5 and y_test[pp]==1):
            TrueP += 1
        if(pred[pp]>0.5 and y_test[pp]==0):
            FalseP += 1
        if(pred[pp]<=0.5 and y_test[pp]==1):
            FalseN += 1
        if(pred[pp]<=0.5 and y_test[pp]==0):
            TrueN += 1
    print("test數量:{}".format(len(x_test)))
    print("True_Positive:{}".format(TrueP))
    print("True_Nagitive:{}".format(TrueN))
    print("False_Positive:{}".format(FalseP))
    print("False_Nagitive:{}".format(FalseN))
    precision = TrueP/(TrueP+FalseP)
    recall = TrueP/(TrueP+FalseN)
    print("Precision:{}".format(precision))
    print("Recall:{}".format(recall))
    # with open("predict.csv", 'a+') as predictcsv:
    #     writer = csv.writer(predictcsv)
    #     writer.writerow([scores1, scores2, scores3])

    # plt.show()