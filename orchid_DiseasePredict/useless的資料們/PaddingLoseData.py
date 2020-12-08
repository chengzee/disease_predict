# orchid data preprocessing
# padding loss data...
# import libraries
import csv
import pandas as pd
import numpy as np

# 讀取氣候資料
sensordataPD = pd.read_csv("sensorNode92.csv")
# print(sensordataPD)
sensordataArray = np.array(sensordataPD)
# print(sensordataArray[0])
# print(len(sensordataArray))
# print(sensordataArray[0][0][11:])
# print(sensordataArray.shape)
# 移除8/19晚上8點以前的資料
# print((sensordataArray[:,0]).shape)
# print(sensordataArray[:,0])
a = 0
label20 = 0
label21 = 0
label22 = 0
label23 = 0
for n in sensordataArray[:, 0]:
    if n[8:10]=="19":
        a+=1
print(a)
# 去掉8/19的斷網前的資療 一切從8/20開始
sensordataArray1 =  sensordataArray[36:]
print(sensordataArray1)
for n in sensordataArray1[:, 0]:
    if n[8:10]=="20":
        label20+=1
    if n[8:10]=="21":
        label21+=1
    if n[8:10]=="22":
        label22+=1
    if n[8:10]=="23":
        label23+=1
print("20:{}, 21:{}, 22:{}, 23:{}".format(label20, label21, label22, label23))
# 有缺，開補
for n in range(len(sensordataArray1[:, 0])):
    if n > 0:
        # 以時間來補
        # ResidualYear = int(sensordataArray1[n][0][0:4])-int(sensordataArray1[n-1][0][0:4])
        # print(ResidualYear)
        # ResidualMonth = int(sensordataArray1[n][0][5:7])-int(sensordataArray1[n-1][0][5:7])
        # print(ResidualMonth)
        # ResidualDay = int(sensordataArray1[n][0][8:10])-int(sensordataArray1[n-1][0][8:10])
        # print(ResidualDay)
        # # print(sensordataArray1[n][0][12:14])
        # ResidualHour = int(sensordataArray1[n][0][12:14])-int(sensordataArray1[n-1][0][12:14])
        # print(ResidualHour)
        # ResidualMinute = int(sensordataArray1[n][0][15:17])-int(sensordataArray1[n-1][0][15:17])
        # print(ResidualMinute)
        # 切割，補上
        # if (ResidualMinute/5)>1:
        #     sensordataArray1[]
        # ------------------------------------------------------------
        # 以傳送編號來補
        # 差幾個編號
        residualNumber = sensordataArray1[n][4]-sensordataArray1[n-1][4]
        if residualNumber>1:
            addvalue = (sensordataArray1[n][1:4]-sensordataArray1[n-1][1:4])/residualNumber
            print(addvalue)
            newdata = np.zeros((residualNumber-1, 5))
            # print(newdata)
            for v in range(residualNumber-1):
                newdata[v][1:4] = sensordataArray1[n-1][1:4] + addvalue*(v+1)
                newdata[v][0] = str(9999)
                newdata[v][4] = 9999
            print(sensordataArray1[0][0])
            print(type(sensordataArray1[0][0]))
            print(newdata)
            sensordataArray1 = np.vstack((np.vstack((sensordataArray1[:n], newdata)), sensordataArray1[n:]))
        sensordataArray1[:, 0] = 0
        np.savetxt('new1.csv', sensordataArray1, delimiter = ',')

            
# print(12/5)
# print(12%7)
# print(int(12/15))