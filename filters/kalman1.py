""" kalman.py

    Contains the iterative kalman filter. But in multiple dimensions :0
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import time

class Kalman():
    ''' Kalman filter implemented within a class object. Multi dimensional :-)

        Arguments:
            var_p: process noise variance, equal for each acceleration axis
            var_m: measurement uncertainty, equal for each acceleration axis
            var_s: initial state uncertainty, equal each for acceleration axis
    '''
    
    # Initialize 
    def __init__(self, var_p, var_m, var_s):
        self.Q = np.array([[var_p, 0, 0], [0, var_p, 0], [0, 0, var_p]]) # process noise variance
        self.R = np.array([[var_m, 0, 0], [0, var_m, 0], [0, 0, var_m]]) # measurement uncertainty
        self.var_s = np.array([[var_s, 0, 0], [0, var_s, 0], [0, 0, var_s]]) # initial state uncertainty
        self.x_hat = np.array([0, 0, 0])
        self.x_hatminus = np.array([0, 0, 0])
        self.K = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.sz = 0
        self.Pminus = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.P = self.var_s # state uncertainty
        self.H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # Observation matrix, just the identity
    
    # Method to update step
    def update_state(self, z):
        self.x_hatminus = self.x_hat
        self.Pminus = self.P + self.Q
        self.K = self.Pminus@np.transpose(self.H)@np.linalg.inv((self.H@self.Pminus@np.transpose(self.H)+self.R))
        self.x_hat = self.x_hatminus + self.K@(np.array(z) - self.H@self.x_hatminus)
        self.P = (np.identity(3) - self.K@self.H)@self.Pminus@np.transpose((np.identity(3)-self.K@self.H)) + self.K@self.R@np.transpose(self.K)
        self.sz += 1
    
    # Return prediction
    def current_state(self):
        return self.x_hat
    
    # Initialize to generic state
    def initialize_state(self, x):
        self.x_hat = x
        

class ExtendedKalman():
    ''' Kalman but nonlinear, write this soon please
    '''
    
# Abstraction of kalman filter to a midflight filter
class MidflightFilter():
    def __init__(self, var_p, var_m, var_s):
        self.kfilter = Kalman(var_p, var_m, var_s)
        self.kfilter.initialize_state([0, 0, 0])
        self.timestamp = time.time()
        
    def process_data(self, data):
        self.kfilter.update_state(data)
        x = self.kfilter.current_state()
        return x
        
class PostFlightFilter():
    def __init__(self):
        pass
        
# A test case
if (__name__ == "__main__"):
    k_filter = Kalman(1e-5, 0.3**2, 2.0)
    k_filter.initialize_state([0.01, 0.01, 0.01])
    while (1):
        z = np.array([1, 1, 1]) + (np.random.rand(1) - np.random.rand(1))*np.random.rand(3)
        k_filter.update_state(z)
        x_filtered = k_filter.current_state()
        time.sleep(1)
        print(f'Noisy measurement:\n {z}\nFiltered Measurement:\n {x_filtered}')