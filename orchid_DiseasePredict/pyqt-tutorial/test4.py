# # import numpy as np
# # # a = [[], [], [], [], [], []]
# # # # print(a.shape)
# # # print(len(a))
# # # b = np.ones((3, 1))
# # # for n in range(6):
# # #     a[n] = b
# # # print(a)
# # # # print(a.shape)
# # # print(len(a))
# # # print(a[0].shape)

# # width, length = (3, 4)
# # humid_thermal_plane_24 = np.zeros((width, length))
# # humid_thermal_plane_24[2, 3] = 69
# # print(humid_thermal_plane_24)
# # # print(13/3)
# # for fs in range(12):
# #     print(fs%3)
# #     print(fs/3)
# #     humid_thermal_plane_24[fs%3, int(fs/3)] = 6699

# # print(humid_thermal_plane_24)
# import numpy as np
# import pandas as pd
# import csv 
# import time
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import os
# # from tensorflow.compat.v1.keras.layers import CuDNNLSTM

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

# save_file_path = './MA5_LuongAtt_byCZ'
# if not os.path.isdir(save_file_path):
#     os.mkdir(save_file_path)

# paddeddata_csv = pd.read_csv("PaddedMovingAverage5_data1.csv") # 讀資料進來
# paddeddata_array = np.array(paddeddata_csv) # 轉為矩陣
# # -------------------------------------------------------------------------------------------------------------------
# # 最小最大值正規化 [0, 1]
# data_min = np.min(paddeddata_array[:, 1:4], axis=0)
# data_max = np.max(paddeddata_array[:, 1:4], axis=0)
# # data_mean = np.mean(paddeddata_array[:, 1:4], axis=0)
# print("data_min:{}".format(data_min))
# print("data_max:{}".format(data_max))
# # # print("data_mean:{}".format(data_mean))
# paddeddata_array_norm = paddeddata_array
# paddeddata_array_norm[:, 1:4] = (paddeddata_array[:, 1:4]-data_min)/(data_max-data_min)
# # print(paddeddata_array[])
# # 參數設定------------------------------------------------------------------------------------------
# count = 1
# the_first_nonzero = 0
# the_last_nonzero = 0
# n = 0
# increase_length = 0
# _lookback = 288+increase_length
# hours = 2
# _delay = 12*hours
# sample_list = []
# target_list = []
# train_size = 0.7
# neurons = [64, 128, 256, 512, 1024]
# # neurons = [1024]
# _source_dim = 3
# predict_dim = 1
# test_times = 6
# BATCH_SIZE = 256
# _epochs = 150
# A_layers = 4

# # 參數設定------------------------------------------------------------------------------------------
# def GenDataset(inputdata, starttime, lasttime, lookback, delay, samp_list, targ_list):
#     for i in range(lasttime-starttime+1):
#         input_raws = np.arange(i+starttime, i+starttime+lookback)
#         output_raws = np.arange(i+starttime+lookback, i+starttime+lookback+delay)
#         samp_list.append(inputdata[input_raws, 1:4])
#         targ_list.append(inputdata[output_raws, 1:4])
#     return samp_list, targ_list
# while 1:
#     if paddeddata_array_norm[n, 4] == 0:
#         the_last_nonzero = n-1
#         count = 1
#         print("creat from {} to {}".format(the_first_nonzero, the_last_nonzero))
#         GenDataset(inputdata=paddeddata_array_norm, starttime=the_first_nonzero, lasttime=the_last_nonzero, lookback=_lookback, delay=_delay, samp_list=sample_list, targ_list=target_list)
#         # check how many zero in next row ~~
#         for p in range(n+1, len(paddeddata_array_norm)):
#             if paddeddata_array_norm[p, 4] == 0: 
#                 count += 1
#             else:
#                 the_first_nonzero = the_last_nonzero + count + 1  
#                 n = the_first_nonzero
#                 break
#     n += 1 
#     if n == len(paddeddata_array_norm):
#         break
# sample_arr = np.array(sample_list)
# target_arr = np.array(target_list)
# print("sample_arr.shape:{}".format(sample_arr.shape))
# print("target_arr.shape:{}".format(target_arr.shape))

# # # # # # # # # # # # # # # # # # # # # # # # # # 
# # -------------------------------------------------------------------------------------------------------------------
# # train test split
# # from sklearn.model_selection import train_test_split
# # x_train, x_test, y_train, y_test = train_test_split(sample_arr, target_arr[:, :, 0], test_size=0.3)
# print("len(sample_arr):{}".format(len(sample_arr)))
# print("len(sample_arr)*train_size:{}".format(len(sample_arr)*train_size))
# x_train = sample_arr[:int(len(sample_arr)*train_size)]
# x_test = sample_arr[int(len(sample_arr)*train_size):]
# y_train = target_arr[:int(len(sample_arr)*train_size), :, 0:predict_dim]
# y_test = target_arr[int(len(sample_arr)*train_size):, :, 0:predict_dim]
# print("x_train.shape:{}".format(x_train.shape))
# print("x_test.shape:{}".format(x_test.shape))
# print("y_train.shape:{}".format(y_train.shape))
# print("y_test.shape:{}".format(y_test.shape))

import numpy as np
da_min = np.array((47.14, 24.66, 0))
print(da_min)