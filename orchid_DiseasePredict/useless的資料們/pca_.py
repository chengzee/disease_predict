import pandas as pd
import numpy as np
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
                X.append(array2[m-1440:m, 1:].reshape((-1)))
                # if array1[n, 8] > -4:
                #     Y.append(0)
                # else:
                #     Y.append(1)
                Y.append(array1[n, 8])

# become array
X = np.array(X, dtype=np.float64)
Y = np.array(Y, dtype=np.float64)
print("X_array's shape:{}".format(X.shape)) # (46, 4320)
print("Y_array's shape:{}".format(Y.shape)) # (46, )
print(X)
# =====================================================================================
from numpy.testing import assert_almost_equal
# Normalization
# 手動正規化
# Z = (X-np.mean(X, axis=0))/np.std(X, axis=0, ddof=0)

# 使用 sklearn API 作正規化 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Z_sk = scaler.fit_transform(X)
print(Z_sk)
print(Z_sk.shape)
# print(assert_almost_equal(Z,Z_sk))
# =====================================================================================
'''
透過 scikit-learn 將 4320 維的氣候資訊降到 2 維
'''
from sklearn.decomposition import PCA
# 只取最大的兩個主成分， scikit-learn 會自動排序出最大的前兩個 eigenvalue 及其對應的 eigenvector
n_components = 2
random_state = 9527
pca = PCA(n_components=n_components, random_state=random_state)
# 對正規化後的特徵 Z 做 PCA
L = pca.fit_transform(Z_sk) # (n_sample, n_components)

import matplotlib.pyplot as plt
# 將投影到第一主成分的 repr. 顯示在 x 軸，第二主成分在 y 軸
plt.figure()
plt.scatter(L[:, 0], L[:, 1], c=Y)
plt.axis('equal') 
# plt.legend()
plt.show()
pca_15d = PCA(15, random_state=random_state)
pca_15d.fit(Z_sk)
print(np.round(pca_15d.explained_variance_ratio_, 3))