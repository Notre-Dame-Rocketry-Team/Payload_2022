from sensors.registers import *
from sensors.mpu_9250 import MPU9250
from sensors.DFRobot_LIS import *
from sensors.bmp280 import *

import smbus,time,datetime

#The I2C address can be switched through the DIP switch (gravity version) or SDO pin (Breakout version) on the board
I2C_BUS         = 0x01            #default use I2C1
#ADDRESS_0       = 0x18            #Sensor address 0
ADDRESS_1       = 0x18            #Sensor address 1 

TIME_LABEL = ['Time']

IMU_LABELS = ["IMU1Acce_X", "IMU1Acce_Y", "IMU1Acce_Z", 
               "IMU1Gyro_X","IMU1Gyro_Y","IMU1Gyro_Z",
               "IMU1Magn_X","IMU1Magn_Y","IMU1Magn_Z", "IMU1Temp"]

ACCE_LABELS = ["Accelerometer X acceleration", "Accelerometer Y acceleration", "Accelerometer Z acceleration"]

BMP_LABELS = ["Altitude"]

def init_time():
    return None, TIME_LABEL

def read_time(_):
    return [time.time()]

def init_imu():
    imu = MPU9250(
        address_ak=AK8963_ADDRESS, 
        address_mpu_master=MPU9050_ADDRESS_68, # Master has 0x68 Address
        address_mpu_slave=MPU9050_ADDRESS_68, # Slave has 0x68 Address
        bus=1, 
        gfs=GFS_1000, 
        afs=AFS_8G, 
        mfs=AK8963_BIT_16, 
        mode=AK8963_MODE_C100HZ)
    imu.configure()
    
    return imu, IMU_LABELS

def read_imu(imu, calibrate=True):
    # each attribute (acceleration,gyro,magnetic) is a tuple (x,y,z) -> Unpacked below.
    acce_imu1 = imu.readAccelerometerMaster() # Acce: g
    gyro_imu1 = imu.readGyroscopeMaster() # Gyro: degrees/s
    magn_imu1 = imu.readMagnetometerMaster() # Magn: μT
    temp_imu1 = imu.readTemperatureMaster() # Temp: Degrees C

    # This is for sensor 1
    m_x = 1.00232407
    b_x = 2.76126821
    m_y = 1.00558483
    b_y = 0.29136013
    m_z = 1.004967481
    b_z = 0.31702996
    
    g_b_x = 1.9915737
    g_b_y = 1.96088371
    g_b_z = -0.67793852
    
    if calibrate:
        acce_imu1[0] = m_x*acce_imu1[0] + b_x
        acce_imu1[1] = m_y*acce_imu1[1] + b_y
        acce_imu1[2] = m_z*acce_imu1[2] + b_z
        
        gyro_imu1[0] = gyro_imu1[0] + g_b_x
        gyro_imu1[1] = gyro_imu1[1] + g_b_y
        gyro_imu1[2] = gyro_imu1[2] + g_b_z
    return acce_imu1, gyro_imu1, magn_imu1, temp_imu1 #, acce_imu2, gyro_imu2, temp_imu2

def init_bmp280():
    bmp = BMP280()
    bmp.setup()
    
    return bmp, BMP_LABELS
    
def read_altitude(bmp):
    altitude = bmp.get_altitude()
    
    return [altitude]
    
def init_accelerometer():
    acce = DFRobot_H3LIS200DL_I2C(I2C_BUS, ADDRESS_1)
    acce.begin()
    acce.set_range(acce.H3LIS200DL_100G)
    acce.set_acquire_rate(0X20) # Set acquire rate of accelerometer manually
    acce.enable_sleep(True)
    acce.enable_int_event(acce.INT_1,acce.Y_HIGHERTHAN_TH)
    time.sleep(1)
    
    return acce, ACCE_LABELS

def read_accelerometer(acce):
    acceXYZ = acce.read_acce_xyz()
    return acceXYZ
    