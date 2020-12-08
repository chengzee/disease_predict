import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.01)
y = np.tanh(x)
plt.figure()
plt.title('Hyperbolic Tangent')
plt.plot(x, y, label='-')
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(-5, 5)
plt.ylim(-1, 1)
plt.show()