import numpy as np
import pandas as pd
import time
import datetime
import csv
# 讀取植床死亡統計資料
rawdatas = pd.read_excel("C:/Users/HUANG/Desktop/modifyV2.xlsx") 
# 轉為 numpy array
rawdatas_array = np.array(rawdatas)
# 輸出「病害死亡csv」，時間以timestamps來表示
with open("target.csv", "w+") as targetfile:
    write = csv.writer(targetfile)
    write.writerow(["Unix Time Stamps", "No.Dead", "No.plant"])
    for n in range(len(rawdatas_array[:, 0])):
        _time = int(time.mktime(time.strptime(str(rawdatas_array[n, 0]), "%Y-%m-%d %H:%M:%S")))
        write.writerow([_time, -rawdatas_array[n, 4], rawdatas_array[n, 7]])
# 讀取「病害死亡csv」
targetRecent = pd.read_csv("target.csv")
# 轉為 numpy array
targetRecent_arr = np.array(targetRecent)

# 改成近期數量
# 以未來三天量為輸出結果
# 當然要先考慮是在同一床的
bed = [631, 742, 701, 759, 765, 698]
for n in bed:
    PlantBed = np.where(targetRecent_arr == n)[0]
    print(PlantBed)
    
    # 計算前後時間差
    for no in range(len(PlantBed)):
        if (no < len(PlantBed)-1) and (0 < targetRecent_arr[PlantBed, 0][no]-targetRecent_arr[PlantBed, 0][no+1] <= 172800):
            if (no < len(PlantBed)-2) and (0 < targetRecent_arr[PlantBed, 0][no]-targetRecent_arr[PlantBed, 0][no+2] <= 172800):
                RecentlyDead = targetRecent_arr[PlantBed, 1][no] + targetRecent_arr[PlantBed, 1][no+1] + targetRecent_arr[PlantBed, 1][no+2]
                targetRecent_arr[PlantBed[no], 1] = RecentlyDead
                print("recentlydead1:{}".format(RecentlyDead))
                print(1)
            else:
                RecentlyDead = targetRecent_arr[PlantBed, 1][no] + targetRecent_arr[PlantBed, 1][no+1]
                targetRecent_arr[PlantBed[no], 1] = RecentlyDead
                print("recentlydead2:{}".format(RecentlyDead))
                print(2)
print("targetRecent_arr:{}".format(targetRecent_arr))
# 輸出「(統計近期三日)近期死亡csv」，時間以timestamps來表示
with open("targetRecent.csv", "w+") as targetRecentfile:
    write = csv.writer(targetRecentfile)
    write.writerow(["Unix Time Stamps", "Recently Dead", "No.plant"])
    # 暫以發生死亡為 Boolean (零或一)
    for n in range(len(targetRecent_arr[:, 0])):
        if targetRecent_arr[n, 1] > 0:
            targetRecent_arr[n, 1] = 1
            write.writerow(targetRecent_arr[n, :])
        else:
            write.writerow(targetRecent_arr[n, :])