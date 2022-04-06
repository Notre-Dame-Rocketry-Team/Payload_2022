import csv
import sensors
import logger
import numpy as np
import time

class dataProcessor():
    ''' Class to process and manipulate data
    '''
    def __init__(self):
        pass
        
    def rotation_constructor(self, gyval, dt):
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        gyval = (np.pi/180.0)*gyval
        
        matrix = np.zeros((3, 3))
        matrix[0, 0] = c(gyval[2])*c(gyval[1])
        matrix[0, 1] = c(gyval[2])*s(gyval[0])*s(gyval[1]) - c(gyval[0])*s(gyval[2])
        matrix[0, 2] = s(gyval[0])*s(gyval[2]) + c(gyval[0])*c(gyval[2])*s(gyval[1])
        matrix[1, 0] = c(gyval[1])*s(gyval[2])
        matrix[1, 1] = c(gyval[0])*c(gyval[2]) + s(gyval[0])*s(gyval[2])*s(gyval[1])
        matrix[1, 2] = c(gyval[0])*s(gyval[2])*s(gyval[1]) - c(gyval[2])*s(gyval[0])
        matrix[2, 0] = -s(gyval[1])
        matrix[2, 1] = c(gyval[1])*s(gyval[0])
        matrix[2, 2] = c(gyval[0])*c(gyval[1])
        
        self.rotation_matrix = matrix
    
    def accel_trans(self, accel, gy_vals): 
        ''' Rotate and change from g to m/s^2
        '''
        a_trans = self.rotation_matrix@accel
        
        return a_trans*9.81
        
class dataFileProcessor():
    ''' Class to read data from the sensor and then save it to a csv file
    
        Args for __init__:
    '''
    def __init__(self, init_functions, read_functions, fname_raw, fname_processed):
        self.init_functions = init_functions
        self.read_functions = read_functions
        self.raw_file = fname_raw
        self.processed_file = fname_processed
        self.sz = 0
        self.dt = 0
        self.orientation = np.array([0.0, 0.0, 0.0])
        self.processor = dataProcessor()

    def initialize_sensors(self):
        out = [f() for f in self.init_functions]
        print(out)
        self.sensors, self.labels = list(zip(*out))
        
    def read_sensors(self):
        self.time1 = time.time()
        self.data = [f(obj) for f, obj in zip(self.read_functions, self.sensors)]
        self.time2 = time.time()
        
        self.dt = self.time2 - self.time1
        
        # Flatten data to one dimensional list
        self.data = [item for sublist in self.data for item in sublist]
        self.data = logger.flatten(self.data)
        self.sz += 1
        
        
    def write_raw(self):
        logger.addRow(self.raw_file, self.data)
        
    def return_raw(self):
        return self.data
        
    def split_data(self):
        self.imu1_a = np.array([self.data[1], self.data[2], self.data[3]])
        self.imu1_gy = np.array([self.data[4], self.data[5], self.data[6]])
        self.orientation += self.imu1_gy*self.dt
    
    def process_data(self):
        self.split_data()
        self.processor.rotation_constructor(self.orientation, self.dt)
        self.imu1_a_f = self.processor.accel_trans(self.imu1_a, self.orientation)
        
    def write_processed(self, state):
        logger.addRow(self.processed_file, [self.time1, state])
        
    def return_processed(self):
        return self.imu1_a_f
    
# THIS ONE IS SPECIFIC TO SUBSCALE DATA, PLEASE KEEP THAT IN MIND
class dataLoader():
    def __init__(self):
        self.time = np.array([])
        self.imu1_a = []
        self.imu1_gy = []
        self.hg_a = []
        self.processor = dataProcessor()
        
    def read_data(self, file):
        firstFlag = 1
        self.sz = 0
        
        with open(file) as data_file:
            data_reader = csv.reader(data_file)
            for row in data_reader:  
                if firstFlag == 1:
                    firstFlag = 0
                    continue
                self.time = np.append(self.time, np.array(float(row[0])))
                self.imu1_a.append([float(row[1]), float(row[2]), float(row[3])])
                self.imu1_gy.append([float(row[4]), float(row[5]), float(row[6])])
                self.sz += 1
        self.imu1_a = np.array(self.imu1_a)
        self.imu1_gy = np.array(self.imu1_gy)
        
        
    def set_time(self):
        ''' 
        Initialize time as 0
        '''
        startTime = self.time[0]
            
        for i in range(len(self.time)):
            self.time[i] -= startTime
                
    def slice_data(self):
        pass

# Testing
if __name__ == "__main__":
    data_loader = dataLoader()
    data_loader.read_data('subscale_data/flight1.csv')
    data_loader.set_time()
    print(f'First Row: \nTime: {data_loader.time[0]}\nAcce: {data_loader.imu1_a[0]}\nGy: {data_loader.imu1_gy[0]}')
     
