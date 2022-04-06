# Import libraries
import sensors
import logging
import glob
import time

# Constants
SAVE_PATH = './data/'
SAVE_NAME = 'data'
SAVE_SUFFIX = '.csv'

INIT_FUNCTIONS = [sensors.init_time,
                  sensors.init_imu,
                  sensors.init_accelerometer]
READ_FUNCTIONS = [sensors.read_time,
                  sensors.read_imu,
                  sensors.read_accelerometer]

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


def initialize_sensors():
    """
    This function calls all of the functions specified in
    INIT_FUNCTIONS to initialize all of the sensors.
    The output from a function in INIT_FUNCTIONS should
    be of the following form:

    Outputs: objects - list of sensor objects
             labels - list of string labels for sensors
    """

    # Run all the functions
    out = [f() for f in INIT_FUNCTIONS]

    # Break it apart
    objects, labels = list(zip(*out))

    return objects, labels


def read_sensors(sensors):
    """
    This function calls all of the functions specified in
    READ_FUNCTIONS to read data from all sensors.
    The output should be a list of all of the sensor data,
    in the order of the label output from `initialize_sensors`

    Inputs: sensors - list of sensor objects
    Outputs: data - list of floats with sensor data
    """

    # Run all the functions
    data = [f(obj) for f, obj in zip(READ_FUNCTIONS, sensors)]

    return data


if __name__ == '__main__':
    # Initialize sensors
    sensors, labels = initialize_sensors()
    fname = find_new_name()
    
    # Initialize CSV
    logging.newCSV(fname, labels)

    while True:
        # Read sensors
        t1 = time.time()
        data = read_sensors(sensors)
        t2 = time.time()
        print(f'Total Sensor Read Time: {t2 - t1}')
        print(data)
        
        # Write to CSV
        t1 = time.time()
        logging.addRow(fname, data)
        t2 = time.time()
        print(f'Data Write Time: {t2 - t1}')
        print()
