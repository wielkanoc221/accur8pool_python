import numpy as np
import matplotlib.pyplot as plt
N=10000
x=np.hanning(N)
print(x)

plt.plot(x)
plt.show()