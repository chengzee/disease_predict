import json
import sys
import requests
import time # 引入time
from datetime import datetime
url = "https://monitor.icmems.ml/api/getDatas/91"
getjsons = requests.get(url).json()
# print(type(getjsons.json()))
# print(type(getjsons['data'][:]))
print(len(getjsons['data'][-10:]))
print(getjsons['data'][-10]['time'])
for n in range(len(getjsons['data'][-10:])):
    timestamp = getjsons['data'][n-10]['time']/1000
    print(timestamp)
    print(type(timestamp))
    # dt = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    dt = datetime.fromtimestamp(int(timestamp))
    print(dt)
    print(type(dt))
    # struct_time = time.localtime(timestamp)
    # print(struct_time)
    # print(type(struct_time))
    # timeString = time.strftime("%Y-%m-%d %H:%M:%S", struct_time) # 轉成字串
    # print(timeString)
    # print(type(datetime.now()))
    # print(getjsons['data'][n-10]['time'])
# print((getjsons['data'][-10:]['temperature']))
# print(len(getjsons))
# print(1)
# print(type(getjsons[0]))


# r = requests.get('https://api.github.com/events')
# print(r)