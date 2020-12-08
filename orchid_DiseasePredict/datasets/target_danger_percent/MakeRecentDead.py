import numpy as np
import pandas as pd
import time
import datetime
import csv
# 每床剩餘數量
each_bed_remain_plants = [239, 307, 308, 309, 314, 301]
# 各床編號
bed = [631, 742, 701, 759, 765, 698]
# 讀取植床死亡統計資料
rawdatas = pd.read_excel("C:/Users/HUANG/Desktop/modifyV3.xlsx") 
# 轉為 numpy array
rawdatas_array = np.array(rawdatas)
# 輸出「病害死亡csv」，時間以timestamps來表示
with open("target.csv", "w+") as targetfile:
    write = csv.writer(targetfile)
    write.writerow(["Unix Time Stamps", "target", "No.plant", "Before_today Remain No."])
    for n in range(len(rawdatas_array[:, 0])):
        _time = int(time.mktime(time.strptime(str(rawdatas_array[n, 0]), "%Y-%m-%d %H:%M:%S")))
        for m in range(len(bed)):
            if rawdatas_array[n, 7] == bed[m]:
                # print("yes")
                write.writerow([_time, -rawdatas_array[n, 4], rawdatas_array[n, 7], each_bed_remain_plants[m]])
# 讀取「病害死亡csv」
targetRecent = pd.read_csv("target.csv")
# 轉為 numpy array
targetRecent_arr = np.array(targetRecent)

targetRecent_arr = targetRecent_arr.astype('float32') 
# print(targetRecent_arr)

# 改成近期數量
# 以在同一床的未來三天淘汰量為輸出結果

#　累積死亡數量
accumulation  = 0

# float(targetRecent_arr)
for n in bed:
    PlantBed = np.where(targetRecent_arr == n)[0]
    print(PlantBed)
    #　累積死亡數量
    accumulation  = 0
    # 計算前後時間差
    for no in range(len(PlantBed)):
        # print(sum(targetRecent_arr[PlantBed, 1][no:]))
        # 當日剩餘
        remain = targetRecent_arr[PlantBed, 3][no]-sum(targetRecent_arr[PlantBed, 1][no:])
        # print(remain)
        # 當天以前
        before_today_remain = targetRecent_arr[PlantBed, 3][no]-sum(targetRecent_arr[PlantBed, 1][no:])+targetRecent_arr[PlantBed, 1][no]
        # print(before_today_remain)
        targetRecent_arr[PlantBed[no], 3] = before_today_remain
        if (no < len(PlantBed)-1) and (0 < targetRecent_arr[PlantBed, 0][no]-targetRecent_arr[PlantBed, 0][no+1] <= 172800):
            if (no < len(PlantBed)-2) and (0 < targetRecent_arr[PlantBed, 0][no]-targetRecent_arr[PlantBed, 0][no+2] <= 172800):
                recently_dead = targetRecent_arr[PlantBed, 1][no] + targetRecent_arr[PlantBed, 1][no+1] + targetRecent_arr[PlantBed, 1][no+2]
                print("recently_dead1:{}".format(recently_dead))
                print("連續一天")
            else:
                recently_dead = targetRecent_arr[PlantBed, 1][no] + targetRecent_arr[PlantBed, 1][no+1]
                # targetRecent_arr[PlantBed[no], 1] = recently_dead
                print("recently_dead2:{}".format(recently_dead))
                print("連續二天")
        else:
            recently_dead = targetRecent_arr[PlantBed, 1][no]
            print("recently_dead0:{}".format(recently_dead))
            print("沒有連續")
        # --------------------------------ｏｐｔｉｏｎ------------------------------------------
        # # 近期死亡數量
        # targetRecent_arr[PlantBed[no], 1] = recently_dead
        # 近期死亡比例
        recently_dead_rate = recently_dead/(targetRecent_arr[PlantBed[no], 3])
        print(recently_dead_rate)
        targetRecent_arr[PlantBed[no], 1] = recently_dead_rate
        print(targetRecent_arr[PlantBed[no]:PlantBed[no]+1, 1:2])
        # --------------------------------------------------------------------------------------
print("targetRecent_arr:{}".format(targetRecent_arr))
print(targetRecent_arr.dtype)
# 輸出「(統計近期三日)近期死亡csv」，時間以timestamps來表示
# with open("targetRecent.csv", "w+") as targetRecentfile:
#     write = csv.writer(targetRecentfile)
#     write.writerow(["Unix Time Stamps", "Recent dead", "No.plant"])
#     # 以發生死亡為 Boolean (零或一)
#     for n in range(len(targetRecent_arr[:, 0])):
#         if targetRecent_arr[n, 1] > 0:
#             targetRecent_arr[n, 1] = 1
#             write.writerow(targetRecent_arr[n, :])
#         else:
#             write.writerow(targetRecent_arr[n, :])

# 輸出「(統計近期三日)近期死亡csv」，時間以timestamps來表示
with open("targetRecent.csv", "w+") as targetRecentfile:
    write = csv.writer(targetRecentfile)
    write.writerow(["Unix Time Stamps", "Recent dead rate", "No.plant"])
    # 以死亡嚴重程度(比例)為 target
    for n in range(len(targetRecent_arr[:, 0])):
        write.writerow(targetRecent_arr[n, :-1])