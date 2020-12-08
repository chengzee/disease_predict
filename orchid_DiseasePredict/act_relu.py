import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-10, 10, 0.01)
# print(type(x))
def relu(input_data, output_data):
    if input_data<=0:
        output_data = 0.0
    else:
        output_data = input_data
    return output_data
y = np.zeros(len(x))
print(len(x))
for i in range(len(x)):
    # print(x[i])
    y[i] = relu(x[i], y[i])
# print(x)
# print(y)
plt.figure()
plt.title('ReLu')
plt.plot(x, y, label='-')
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-5, 5)
plt.ylim(-1, 5)
plt.show()
# # x = np.arange(-10, 10)
# y = np.zeros(20)
# plt.figure()
# plt.plot(x, y, label='-')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.xlim(-5, 5)
# plt.ylim(-1, 5)
# plt.show()
