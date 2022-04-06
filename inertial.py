# Import libraries
#import sensor as s
import data_read as dr
import glob
import time
import filters.kalman1 as kalman
import matplotlib.pyplot as plt
import numpy as np

loader = dr.dataLoader()
loader.read_data('subscale_data/flight1.csv')
loader.set_time()

filtered_data = []
mid_filter = kalman.MidflightFilter(1e-5, 0.3**2, 2.0)

for i in range(loader.sz):
    x_filter = [mid_filter.process_data(loader.imu1_a[i])]
    filtered_data.append(x_filter)

filtered_data = np.array(filtered_data)
print(filtered_data.shape)
y = filtered_data[:,0,1]
print(y[1])

x = loader.time
plt.plot(x, y, 'r')
plt.plot(loader.imu1_a[:, 1], linewidth=1)
plt.show()