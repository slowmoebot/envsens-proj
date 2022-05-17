import numpy as np
import matplotlib.pyplot as plt

arr=np.load("vars/alldata_da2-est_uncert-P100-w000_CO.npy")
print(arr)

x = np.arange(0,arr.shape[0])
y = arr[:,2]
error = arr[:,5]

plt.fill_between(x, y-error, y+error)
plt.plot(x, y,"k-")
plt.show()