import numpy as np

class DetectState:
    def __init__(self):
        
        self.a_queue = []
        self.g_queue = []
        self.size = 0
        self.acell_tolerance_land = .3
        self.gyro_tolerance_land = 5
        self.acell_tolerance_launch = 10
        
    def update(self, a_array, g_array):
        if self.size >= 10:
            self.a_queue.pop(0)
            self.g_queue.pop(0)
            self.size -=1
        self.a_queue.append(a_array)
        self.g_queue.append(g_array)
        self.size += 1

    def calc_var(self):
        accel = np.array(self.a_queue)
        gyro = np.array(self.g_queue)
        
        self.acell_std = np.std(accel, axis=0)
        self.gyro_std = np.std(gyro, axis=0)

    def checkLanding(self):
        self.calc_var()
        if np.linalg.norm(self.acell_std) < self.acell_tolerance_land and np.linalg.norm(self.gyro_std) < self.gyro_tolerance_land:
            return True
        else:
            return False
		   
		   
class DetectLaunch():
    def __init__(self):
        
        self.a_queue = []
        self.size = 0
        self.accel_tolerance_launch = 2
        
    def update(self, a_array):
        if self.size >= 2:
            self.a_queue.pop(0)
            self.size -=1
        self.a_queue.append(a_array)
        self.size += 1

    def calc_avg(self):
        accel = np.array(self.a_queue)
        self.accel_avg = np.average(accel, axis=0)
        
    def checkLaunch(self):
        self.calc_avg()
        if( abs(np.amax(self.accel_avg)) > self.accel_tolerance_launch ) :
            return True
        else:
            return False