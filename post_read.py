import data_read as dr
import matplotlib.pyplot as plt
from filters.kalman3 import ExtendedKalman

import numpy as np

def post_flight_analysis(data_file, orientation):
    loader = dr.dataLoader()
    loader.read_data(data_file)
    loader.set_time()

    acce = loader.imu1_a
    gyro = loader.imu1_gy
    times = loader.time

    init_state = np.zeros(12)
    init_state[9] += orientation[0]
    init_state[10] += orientation[1]
    init_state[11] += orientation[2]
    
    pf_filter = ExtendedKalman(1e-3, 4.0, 0.1, 0.5)
    pf_filter.initialize_state(init_state)

    states = []
    N = loader.imu1_a.shape[0]
    for i in range(N):
        if i == 0:
            dt = 0
        else:
            dt = times[i] - times[i-1]
        pf_filter.update_state(9.8*acce[i], gyro[i]/180.0*np.pi, dt)
        current_state = pf_filter.current_state().tolist()
        states.append(current_state)
    
    position = np.array([current_state[0], current_state[3], current_state[6]])
    return position