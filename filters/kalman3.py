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
        
        self.R = np.array([[var_m_a**2, 0, 0], 
                           [0, var_m_a**2, 0], 
                           [0, 0, var_m_a**2]]) # measurement uncertainty
        
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
        for i in range(12):
            self.P[i, i] = var_s
            
        # Initialize observation Jacobian
        self.J_h = self.get_H_j()
        
    def Ffunc(self, state, gyro, dt):
        ''' The transformation function
        
        x_k = Ffunc(x_k-1, u_k)
        
        where in this case x_k-1 is state and u_k is gyro
        '''
        
        # Find changes in Euler angles using gyroscope
        du = gyro[0]*dt
        dv = gyro[1]*dt
        dw = gyro[2]*dt
        
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        x_k = np.zeros(12)
        
        x_0 = state[0]
        v_x0 = state[1]
        a_x0 = state[2]
        y_0 = state[3]
        v_y0 = state[4]
        a_y0 = state[5]
        z_0 = state[6]
        v_z0 = state[7]
        a_z0 = state[8] 
        
        u = state[9]
        v = state[10]
        w = state[11]
        
        x_k[0] = x_0 + v_x0*dt + a_x0* dt**2 / 2    # x = x_0 + v_x*dt + a_x*dt^2 / 2
        x_k[1] = c(dw)*c(dv)*v_x0 + ( c(dw)*s(dv)*s(du) - s(dw)*c(du) )*v_y0 + ( c(dw)*s(dv)*c(du) + s(dw)*s(du) )*v_z0 + a_x0*dt
        x_k[2] = c(dw)*c(dv)*a_x0 + ( c(dw)*s(dv)*s(du) - s(dw)*c(du) )*a_y0 + ( c(dw)*s(dv)*c(du) + s(dw)*s(du) )*a_z0
        
        x_k[3] = y_0 + v_y0*dt + a_y0* dt**2 / 2
        x_k[4] = s(dw)*c(dv)*v_x0 + ( s(dw)*s(dv)*s(du) + c(dw)*c(du) )*v_y0 + ( s(dw)*s(dv)*c(du) - c(dw)*s(du) )*v_z0 + a_y0*dt
        x_k[5] = s(dw)*c(dv)*a_x0 + ( s(dw)*s(dv)*s(du) + c(dw)*c(du) )*a_y0 + ( s(dw)*s(dv)*c(du) - c(dw)*s(du) )*a_z0
        
        x_k[6] = z_0 + v_z0*dt + a_z0* dt**2 / 2
        x_k[7] = -s(dv)*v_x0 + ( c(dv)*s(du) )*v_y0 + ( c(dv)*c(du) )*v_z0 + a_z0*dt
        x_k[8] = -s(dv)*a_x0 + ( c(dv)*s(du) )*a_y0 + ( c(dv)*c(du) )*a_z0
        
        x_k[9] = u + du
        x_k[10] = v + dv 
        x_k[11] = w + dw
        
        return x_k
        
    def get_H_f(self, gyro, dt):
        ''' Get the Jacobian approximating the transformation function J_f
        '''
        
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        du = gyro[0]*dt
        dv = gyro[1]*dt
        dw = gyro[2]*dt
        
        J_f = np.array([[1, dt, dt**2/2, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                       [0, c(dw)*c(dv), dt, 0, c(dw)*s(dv)*s(du) - s(dw)*c(du), 0, 0, c(dw)*s(dv)*c(du) + s(dw)*s(du), 0, 0, 0, 0],
                       [0, 0, c(dw)*c(dv), 0, 0, c(dw)*s(dv)*s(du) - s(dw)*c(du), 0, 0, c(dw)*s(dv)*c(du) + s(dw)*s(du), 0, 0, 0],
                       [0, 0, 0, 1, dt, dt**2/2, 0, 0, 0, 0, 0, 0],
                       [0, s(dw)*c(dv), 0, 0, s(dw)*s(dv)*s(du) + c(dw)*c(du), dt, 0, s(dw)*s(dv)*c(du) - c(dw)*s(du), 0, 0, 0, 0],
                       [0, 0, s(dw)*c(dv), 0, 0, s(dw)*s(dv)*s(du) + c(dw)*c(du), 0, 0, s(dw)*s(dv)*c(du) - c(dw)*s(du), 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, dt, dt**2/2, 0, 0, 0],
                       [0, -s(dv), 0, 0, c(dv)*s(du), 0, 0, c(dv)*c(du), dt, 0, 0, 0],
                       [0, 0, -s(dv), 0, 0, c(dv)*s(du), 0, 0, c(dv)*c(du), 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
        
        return J_f
        
    def hfunc(self, state):
        ''' The observation function 
        
        z = h(x_)
        
        where z is the measurement vector and x_ is the state of the system vector
        '''
        
        z = np.zeros(3)
        
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
                      [0, 0, -s(v), 0, 0, c(v)*s(u), 0, 0, c(v)*c(u), az_u, az_v, az_w]])
        
        return J
        
    # Correct for gravity
    def correct_z(self, z):
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        g = 9.8
        
        u = -self.x_hatminus[9]
        v = -self.x_hatminus[10]
        w = -self.x_hatminus[11]
        
        ax_s = z[0]
        ay_s = z[1]
        az_s = z[2]
        
        z_new = np.zeros(3)
        
        z_new[0] = c(w)*c(v)*ax_s + ( c(w)*s(v)*s(u) - s(w)*c(u) )*ay_s + ( c(w)*s(v)*c(u) + s(w)*s(u) )*az_s
        z_new[1] = s(w)*c(v)*ax_s + ( s(w)*s(v)*s(u) + c(w)*c(u) )*ay_s + ( s(w)*s(v)*c(u) - c(w)*s(u) )*az_s
        z_new[2] = -s(v)*ax_s + ( c(v)*s(u) )*ay_s + ( c(v)*c(u) )*az_s - g
        
        z_corr = np.zeros(3)
        
        ax_s = z_new[0]
        ay_s = z_new[1]
        az_s = z_new[2]
        
        u *= -1
        v *= -1
        w *= -1
        
        z_corr[0] = c(w)*c(v)*ax_s + ( c(w)*s(v)*s(u) - s(w)*c(u) )*ay_s + ( c(w)*s(v)*c(u) + s(w)*s(u) )*az_s
        z_corr[1] = s(w)*c(v)*ax_s + ( s(w)*s(v)*s(u) + c(w)*c(u) )*ay_s + ( s(w)*s(v)*c(u) - c(w)*s(u) )*az_s
        z_corr[2] = -s(v)*ax_s + ( c(v)*s(u) )*ay_s + ( c(v)*c(u) )*az_s
        
        return z_corr

    # Method to update step
    def update_state(self, z, gyro, dt):
        ''' z in this case is acceleration data and gyro is (du/dt, dv/dt, and dw/dt), i.e. the readings from the
        gyroscope
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
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1*self.var_m_g, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1*self.var_m_g, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1*self.var_m_g]])
        
        # Extrapolate next state
        self.x_hatminus = self.Ffunc(self.x_hat, gyro, dt)
        
        # Get J_f matrix
        self.F = self.get_H_f(gyro, dt)
        
        # Extrapolate uncertainty
        self.Pminus = self.F@self.P@np.transpose(self.F) + self.Q
        
        # Recalculate Jacobian
        self.J_h = self.get_H_j()
        
        # Calculate Kalman gain
        self.K = self.Pminus@ np.transpose(self.J_h) @ np.linalg.inv(self.J_h @ self.Pminus @ np.transpose(self.J_h) + self.R)
        
        # Correct z for gravity
        z = self.correct_z(z)
        
        # Correct estimation using measurement
        print(f'Observation trans: {self.hfunc(self.x_hatminus)}')
        print(f'Corrected z: {z}')
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
    #   Extended Test Case           #
    ##################################
    
    pf_filter = ExtendedKalman(1e-4, 5.0, 3.0, 10)
    pf_filter.initialize_state([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_estimate_states = []
    y_estimate_states = []
    z_estimate_states = []
    true_states= []
    times = []
    t = 0
    
    for i in range(600):
        w_rate = 1.0
        
        z = np.array([0.0, 0, 1]) + (np.random.rand(1) - np.random.rand(1))*np.random.rand(3)/6
        z = z*9.8
        gyro = np.array([0, 0, w_rate]) #+  (np.pi/180.0)*(np.random.rand(1) - np.random.rand(1))*np.random.rand(3)/8
        dt = 0.2
       
        pf_filter.update_state(z, gyro, dt)
        state = pf_filter.current_state()
        
        print(f'Measured z: {z}')
        print(f'Measured gyro: {gyro}')
        print(state)
        x_estimate_states.append(state[0])
        y_estimate_states.append(state[3])
        z_estimate_states.append(state[6])
        
        times.append(t)
        
        t += dt
        
    print(f'Time: {t}')
    
    # convert to np arrays
    x_estimate_states = np.array(x_estimate_states)
    y_estimate_states = np.array(y_estimate_states)
    z_estimate_states = np.array(z_estimate_states)
    
    times = np.array(times)
    
    # Plotting
    figure1 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_estimate_states, y_estimate_states, z_estimate_states, label="Filtered Measurements")
    ax.axes.set_xlim3d(-5, 5)
    ax.axes.set_ylim3d(-5, 5)
    ax.axes.set_zlim3d(-2, 5)
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.show()