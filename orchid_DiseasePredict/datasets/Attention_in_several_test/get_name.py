import os
list1 = []
path='./'
files = os.listdir(path)
for file in files:
    if file[-3:] == ".py":
        print(file)
