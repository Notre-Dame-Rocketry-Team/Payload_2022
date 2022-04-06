''' Master pi
    pi@rocknroll
    pi@theking
'''

# Import libraries
import sensor as s
import data_read as dr
import glob
import time
import filters.kalman3 as kalman
import numpy as np
import logger as logging
import post_read

from grid_finder import get_position
from ahrs.filters import SAAM
import state_detector as sd
import orientation as ori

# Constants
SAVE_PATH = './data/'
SAVE_NAME = 'data'
SAVE_SUFFIX = '.csv'

ALTITUDE_TOLERANCE = 5

INIT_FUNCTIONS = [s.init_time,
                  s.init_imu,
                  s.init_bmp280,
                  s.init_accelerometer]
READ_FUNCTIONS = [s.read_time,
                  s.read_imu,
                  s.read_altitude,
                  s.read_accelerometer]

def find_new_name():
    # Step 1: search for file names using glob
    files = glob.glob(SAVE_PATH + SAVE_NAME + "_*" + SAVE_SUFFIX)

    # Step 2: if there are any file names, find the biggest number
    numbers = []

    for x in files: 
        numbers.append(int(x.replace(SAVE_PATH + SAVE_NAME + "_", "").replace(SAVE_SUFFIX, "")))

    z = 1 + max(numbers) if len(numbers) > 0 else 0

    # Step 3: Return the output file name
    fname = SAVE_PATH + SAVE_NAME + '_' + str(z) + SAVE_SUFFIX

    return fname

def find_new_name_processed():
    # Step 1: search for file names using glob
    files = glob.glob(SAVE_PATH + SAVE_NAME + "_*" + SAVE_SUFFIX)

    # Step 2: if there are any file names, find the biggest number
    numbers = []

    for x in files: 
        numbers.append(int(x.replace(SAVE_PATH + SAVE_NAME + "_", "").replace(SAVE_SUFFIX, "")))

    z = 1 + max(numbers) + 1 if len(numbers) > 0 else 1

    # Step 3: Return the output file name
    fname = SAVE_PATH + SAVE_NAME + '_' + str(z) + SAVE_SUFFIX

    return fname
    
def test_protocol(processor, k_filter, r):
    #Function to test reading, 'riting, 'rithmetic (conversion from body frame to absolute frame)
        # Read sensors
        processor.read_sensors()
        data = processor.return_raw()
        processor.split_data()
        print(data)
        
        
    
if __name__ == '__main__':
    
    r_ = np.zeros(9)
    
    # Find new file name to save to
    fname = find_new_name()
    fname_p = find_new_name()
    
    # Start processor and initialize sensors
    processor = dr.dataFileProcessor(INIT_FUNCTIONS, READ_FUNCTIONS, fname, fname_p)
    processor.initialize_sensors()
    
    # Initialize CSV
    logging.newCSV(fname, processor.labels)
    
    # Initialize filter
    mid_filter = kalman.MidflightFilter(r_, 1e-3, 0.8**2, 4.0)
    
    # Initialize launch and landing detectors
    launch_checker = sd.DetectLaunch()
    land_checker = sd.DetectState()
    
    # Initialize initial orientation estimator
    saam = SAAM()
    
    '''
    while True:
        try:
            test_protocol(processor, mid_filter, r_)
        except KeyboardInterrupt:
            break
            
    '''
    
    # Determine init_altitude
    processor.read_sensors()
    data = processor.return_raw()
    
    init_altitude = data[11]
    cond = False
    while True:
        try:
            processor.read_sensors()
            data = processor.return_raw()
            
            launch_checker.update(data[1:4])
            
            if launch_checker.size >= 10:
                cond = launch_checker.checkLanding()
                
            if cond and altitude > init_altitude+ALTITUDE_TOLERANCE:
                break
            
            orientation = saam.estimate(acc=data[1:4], mag=data[7:10])
        except KeyboardInterrupt:
            break
    
    euler_orientation = ori.Orientation()
    
    orientation = euler_orientation.Quaternion2Angle(orientation)
    print(f'Final Orientation from Initial in Terms of Euler Angles: {orientation}')
    
    print(f'Reading')
    cond = False
    while True:
        try:
            processor.read_sensors()
            data = processor.return_raw()
            
            processor.write_raw()
            
            land_checker.update(data[1:4], data[4:7])
            altitude = data[11]
            if land_checker.size >= 2:
                cond = land_checker.checkLanding()
            
            if cond and altitude < init_altitude + ALTITUDE_TOLERANCE:
                print(cond)
                break
            
        except KeyboardInterrupt:
            break
    
    print(f'Reading stopped; flight ended')
    print(f'Running final position calculations...')
    final_position = post_read.post_flight_analysis(fname, orientation)
    
    print(f'Final position: {final_position}')
    
    # Convert to feet
    final_position = 3.28084*final_position
    
    # Calculate
    grid_num = get_position(final_position[0], final_position[1])
    
    # Transmit
    
