""" kalman.py

    Contains the iterative kalman filter. But in multiple dimensions :0
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import time

class Kalman():
    ''' Implementation of the Kalman Filter. Multi dimensional :-)

        Arguments:
            var_p: process noise variance, equal for each acceleration axis
            var_m: measurement uncertainty, equal for each acceleration axis
            var_s: initial state uncertainty, equal each for acceleration axis
            
        This specific implementation has the state of the system be a nine dimensional vector, x, xdot, xdoubledot, ... , z, zdot, zdoubledot
        i.e., where the first entry is position x, the second entry is x component of velocity, the third entry is the x component of acceleration, and
        the repeat for y and z. The measurements it uses to estimate of the system is accelerometer data (a_x, a_y, a_z). 
    '''
    
    # Initialize 
    def __init__(self, var_p, var_m, var_s):
        self.var_p = var_p
        self.var_m = var_m
        self.var_s = var_s
        
        self.R = np.array([[var_m**2, 0, 0], [0, var_m**2, 0], [0, 0, var_m**2]]) # measurement uncertainty
        
        # Initial states
        self.x_hat = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.x_hatminus = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Initial Kalman gain
        self.K = np.zeros((9, 9))
        
        # Initial size of dataset
        self.sz = 0
        
        # Initial covariances
        self.Pminus = np.zeros((9, 9))
        self.P = np.zeros((9, 9))
        
        # Set diagonal entries of covariance to var_s
        for i in range(9):
            self.P[i, i] = var_s
        
        # transformation matrix from state to measured, i.e.   z = H @ x, where x is a vector of the stae of the matrix
        self.H = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0], 
                           [0, 0, 0, 0, 0, 1, 0, 0, 0], 
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]]) 
    
    # Method to update step
    def update_state(self, z, dt):
        ''' z in this case is acceleration data
        '''
        # Process noise variance
        self.Q = self.var_p * np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0, 0, 0, 0],
                           [dt**3/2, dt**2, dt, 0, 0, 0, 0, 0, 0],
                           [dt**2/2, dt, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                           [0, 0, 0, dt**3/2, dt**2, dt, 0, 0, 0],
                           [0, 0, 0, dt**2/2, dt, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
                           [0, 0, 0, 0, 0, 0, dt**3/2, dt**2, dt],
                           [0, 0, 0, 0, 0, 0, dt**2/2, dt, 1]])
        
        # State transition
        self.F = np.array([[1, dt, dt**2/2, 0, 0, 0, 0, 0, 0],
                           [0, 1, dt, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, dt, dt**2/2, 0, 0, 0],
                           [0, 0, 0, 0, 1, dt, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, dt, dt**2/2],
                           [0, 0, 0, 0, 0, 0, 0, 1, dt],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1]])
                           
        # Extrapolate state
        self.x_hatminus = self.F@self.x_hat
        
        # Extrapolate uncertainty
        self.Pminus = self.F@self.P@np.transpose(self.F) + self.Q
        
        # Calculate Kalman gain
        self.K = self.Pminus@np.transpose(self.H)@ np.linalg.inv((self.H@self.Pminus@np.transpose(self.H)+self.R))
        
        # Correct estimation using measurement
        self.x_hat = self.x_hatminus + self.K@(np.array(z) - self.H@self.x_hatminus)
        
        # Update estimate uncertainty
        self.P = (np.identity(9) - self.K@self.H)@self.Pminus@np.transpose((np.identity(9)-self.K@self.H)) + self.K@self.R@np.transpose(self.K)
        
        self.sz += 1 
        
    # Return prediction
    def current_state(self):
        return self.x_hat
    
    # Initialize to generic state
    def initialize_state(self, x):
        self.x_hat = x
        

# Abstraction of kalman filter to a midflight filter
class MidflightFilter():
    def __init__(self, state_init, var_p, var_m, var_s):
        self.kfilter = Kalman(var_p, var_m, var_s)
        self.kfilter.initialize_state(state_init)
        self.timestamp = time.time()
        
    def process_data(self, state, data, dt):
        ''' Data  in this case is a np array, where first element is a_x, second is a_y, and third
        is a_z
        
        State is nine dimensional array of the n-1 state of the system
        '''
        
        self.kfilter.update_state(data, dt)
        x = self.kfilter.current_state()
        return x

class ExtendedKalman():
    ''' In this specific implementation of the EKF, the state vector is (x, vx, ax, y, vy, ay, z, vz, az, u, v, w), where
    u, v, and w are the Euler angles describing the vehicle's orientation with respect to a space reference frame
    '''
    def __init__(self, var_p, var_m_a, var_m_g, var_s):
        self.var_p = var_p
        self.var_m_a = var_m_a
        self.var_s = var_s
        self.var_m_g = var_m_g
        
        self.R = np.array([[var_m_a**2, 0, 0, 0, 0, 0], 
                           [0, var_m_a**2, 0, 0, 0, 0], 
                           [0, 0, var_m_a**2, 0, 0, 0],
                           [0, 0, 0, var_m_g**2, 0, 0],
                           [0, 0, 0, 0, var_m_g**2, 0],
                           [0, 0, 0, 0, 0, var_m_g**2]]) # measurement uncertainty
        
        # Initial states
        self.x_hat = np.zeros(12)
        self.x_hatminus = np.zeros(12)
        
        # Initial Kalman gain
        self.K = np.zeros((12, 12))
        
        # Initial size of dataset
        self.sz = 0
        
        # Initial covariances
        self.Pminus = np.zeros((12, 12))
        self.P = np.zeros((12, 12))
        
        # Set diagonal entries of covariance to var_s
        for i in range(9):
            self.P[i, i] = var_s
            
        # Initialize observation Jacobian
        self.J_h = self.get_H_j()
    
    def hfunc(self, state):
        ''' The observation function 
        
        z = h(x_)
        
        where z is the measurement vector and x_ is the state of the system vector
        '''
        
        z = np.zeros(6)
        
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        ax_s = state[2]
        ay_s = state[5]
        az_s = state[8]
        
        u = state[9]
        v = state[10]
        w = state[11]
        
        z[0] = c(w)*c(v)*ax_s + ( c(w)*s(v)*s(u) - s(w)*c(u) )*ay_s + ( c(w)*s(v)*c(u) + s(w)*s(u) )*az_s
        z[1] = s(w)*c(v)*ax_s + ( s(w)*s(v)*s(u) + c(w)*c(u) )*ay_s + ( s(w)*s(v)*c(u) - c(w)*s(u) )*az_s
        z[2] = -s(v)*ax_s + ( c(v)*s(u) )*ay_s + ( c(v)*c(u) )*az_s
        z[3] = u
        z[4] = v
        z[5] = w
        
        return z
        
    def get_H_j(self):
        ''' Get the Jacobian of the observation function
        
        u: roll
        v: pitch
        w: yaw
        Sweet-faced ones with nothing left inside
        
        J is the Jacobian where the rows span over the measurement variables 
        
        ax
        ay
        az
        u
        v
        w
        
        and the columns span over the entries of state vector. I.e., the 0, 0 entry is
        (d/dx)*(ax)
        the 0, 11 entry is
        (d/dw)*(ax)
        and the 11, 11 entry is
        (d/dw)*(w)
        Does this make sense? My brain is gloop. To be good about, here is the h transformation function.
        
        ax = c(u)*c(v)*xddot 
        '''
        
        # Turn negative is ax, ay, az are the acceleration in the body frame, while the acceleration in the
        # state vector is acceleration in the space frame
        u = self.x_hatminus[9]
        v = self.x_hatminus[10]
        w = self.x_hatminus[11]
        
        # Get acceleration in space reference frame
        ax_s = self.x_hatminus[2]
        ay_s = self.x_hatminus[5]
        az_s = self.x_hatminus[8]
        
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        # Calculate Jacobian components with respect to u, v, and w separately for readibility?. OMG have i actually finished?
        ax_u = s(u)*( ay_s * s(w) - az_s * s(v) * c(w)) + c(u)*( ay_s * s(v) * c(w) + az_s * s(w) )
        ax_v = c(w)*( ay_s*s(u)*c(v) + az_s*c(u)*c(v) - ax_s*s(v) ) 
        ax_w = -s(w)*( ay_s*s(u)*s(v) + ax_s*c(v)) - c(u)*(az_s*s(v)*s(w) + ay_s*c(w)) + az_s*s(u)*c(w)
        
        ay_u = c(u)*( ay_s*s(v)*s(w) - az_s*c(w)) - s(u)*( az_s*s(v)*s(w) + ay_s*c(w) )
        ay_v = s(w)*( ay_s*s(u)*c(v) + az_s*c(u)*c(v) - ax_s*s(v))
        ay_w = c(u)*( az_s*s(v)*c(w) - ay_s*s(w) ) + s(u)*( ay_s*s(v)*c(w) + az_s*s(w) ) + ax_s*c(v)*c(w)
        
        az_u = c(v)*( ay_s*c(u) - az_s*s(u))
        az_v = -s(v)*( ay_s*s(u) + az_s*c(u)) - ax_s*c(v)
        az_w = 0
        
        J = np.array([[0, 0, c(w)*c(v), 0, 0, c(w)*s(v)*s(u) - s(w)*c(u), 0, 0, c(w)*s(v)*c(u) + s(w)*s(u), ax_u, ax_v, ax_w], 
                      [0, 0, s(w)*c(v), 0, 0, s(w)*s(v)*s(u) + c(w)*c(u), 0, 0, s(w)*s(v)*c(u) - c(w)*s(u), ay_u, ay_v, ay_w],
                      [0, 0, -s(v), 0, 0, c(v)*s(u), 0, 0, c(v)*c(u), az_u, az_v, az_w],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        return J
        
    # Method to update step
    def update_state(self, z, dt):
        ''' z in this case is acceleration data
        '''
        # Process noise variance
        self.Q = self.var_p * np.array([[dt**4/4, dt**3/2, dt**2/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [dt**3/2, dt**2, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [dt**2/2, dt, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, dt**4/4, dt**3/2, dt**2/2, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, dt**3/2, dt**2, dt, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, dt**2/2, dt, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, dt**3/2, dt**2, dt, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, dt**2/2, dt, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        # State transition
        self.F = np.array([[1, dt, dt**2/2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, dt, dt**2/2, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, dt, dt**2/2, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
                           
        # Extrapolate next state
        self.x_hatminus = self.F@self.x_hat
        
        # Extrapolate uncertainty
        self.Pminus = self.F@self.P@np.transpose(self.F) + self.Q
        
        # Recalculate Jacobian
        self.J_h = self.get_H_j()
        
        # Calculate Kalman gain
        self.K = self.Pminus@ np.transpose(self.J_h) @ np.linalg.inv(self.J_h @ self.Pminus @ np.transpose(self.J_h) + self.R)
        
        # Correct estimation using measurement
        self.x_hat = self.x_hatminus + self.K@(np.array(z) - self.hfunc(self.x_hatminus))
        
        # Update estimate uncertainty
        self.P = (np.identity(12) - self.K@self.J_h)@self.Pminus
        
        self.sz += 1 
        
    # Return prediction
    def current_state(self):
        return self.x_hat
        
    # Initialize to generic state
    def initialize_state(self, x):
        self.x_hat = x
        
# A test case
if (__name__ == "__main__"):

    ##################################
    #  Linear Test Case              #
    ##################################
    '''
    # Test example, filter very noisy measurements measuring a system at rest
    measurements = []
    state = []
    times = []
    
    # Initialize filter
    t = 0
    k_filter = Kalman(5e-5, 0.2, 10)
    k_filter.initialize_state([0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # Do the measurements 100 times
    for i in range(100):
        z = np.array([0, 0, 0]) + (np.random.rand(1) - np.random.rand(1))*np.random.rand(3)/4
        k_filter.update_state(z, 0.1)
        x_filtered = k_filter.current_state()
        print(f'Noisy measurement:\n {z}\nFiltered Measurement:\n {x_filtered}\n')
        measurements.append(z[0])
        state.append((k_filter.H @ x_filtered)[0])
        times.append(t)
        t += 0.1
    
    # convert to np arrays
    state = np.array(state)
    measurements = np.array(measurements)
    times = np.array(times)
    
    # Plotting
    figure1 = plt.figure()
    plt.plot(times, measurements, label="Noisy Measurements")
    plt.plot(times, state, label="Filtered Measurements")
    plt.plot(times, np.zeros(k_filter.sz), label="True State of the System")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s)")
    plt.legend()
    
    plt.ylim([-0.3, 0.3])
    plt.show()
    '''
    
    ##################################
    #   Extended Test Case           #
    ##################################
    
    pf_filter = ExtendedKalman(1e-3, 0.2, 0.5, 10)
    pf_filter.initialize_state([0, 3, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0])
    x_estimate_states = []
    y_estimate_states = []
    z_estimate_states = []
    true_states= []
    times = []
    t = 0
    
    for i in range(50):
        w_rate = 30
        z = np.array([0, 0, -9.8, 0, 0, w_rate*i]) + (np.random.rand(1) - np.random.rand(1))*np.random.rand(6)/4
        
        dt = 0.01
        
        pf_filter.update_state(z, dt)
        state = pf_filter.current_state()
        
        x_estimate_states.append(state[0])
        y_estimate_states.append(state[3])
        z_estimate_states.append(state[6])
        
        times.append(t)
        
        t += dt
        
    # convert to np arrays
    x_estimate_states = np.array(x_estimate_states)
    y_estimate_states = np.array(y_estimate_states)
    z_estimate_states = np.array(z_estimate_states)
    
    times = np.array(times)
    
    # Plotting
    figure1 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_estimate_states, y_estimate_states, z_estimate_states, label="Filtered Measurements")
    #plt.plot(times, np.zeros(k_filter.sz), label="True State of the System")
    ax.axes.set_xlim3d(0, 2)
    ax.axes.set_ylim3d(0, 2)
    ax.axes.set_zlim3d(0, 2)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.show()