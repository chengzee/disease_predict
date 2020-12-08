import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.01)
y = 1/(1+np.exp(-x))
# plt.figure()
# plt.plot(x, y, label='-')
# plt.show()
plt.figure()
plt.title('Sigmoid')
plt.plot(x, y, label='-')
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-5, 5)
plt.ylim(0, 1)
plt.show()