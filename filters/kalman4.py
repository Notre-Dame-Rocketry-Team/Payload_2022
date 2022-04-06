'''
When I wrote this code only god and I knew how it worked. Mow only God knows. Goodluck asshole.
'''


import numpy as np
import matplotlib.pyplot as plt

class ExtendedKalman():
    ''' In this specific implementation of the EKF, the state vector is (x, vx, ax, y, vy, ay, z, vz, az, u, v, w), where
    u, v, and w are the Euler angles describing the vehicle's orientation with respect to a space reference frame
    '''
    def __init__(self, var_p, var_m_a, var_m_g, var_s):
        self.var_p = var_p
        self.var_m_a = var_m_a
        self.var_s = var_s
        self.var_m_g = var_m_g
        
        self.R = np.array([[var_m_a**2, 0], 
                           [0, var_m_a**2]]) # measurement uncertainty
        
        # Initial states
        self.x_hat = np.zeros(9)
        self.x_hatminus = np.zeros(9)
        
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
            
        # Initialize observation Jacobian
        self.J_h = self.get_H_j()
        
    def Ffunc(self, state, acce, gyro, dt):
        ''' The transformation function
        
        x_k = Ffunc(x_k-1, u_k)
        
        where in this case x_k-1 is state and u_k is gyro
        '''
        
        
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        x_k = np.zeros(9)
        
        # Get position + velocities in space frame
        x_0 = state[0]
        v_x0 = state[1]
        y_0 = state[2]
        v_y0 = state[3]
        z_0 = state[4]
        v_z0 = state[5]
        
        # Get orientation
        u_old = state[6]
        v_old = state[7]
        w_old = state[8]
        
        # Get accelerations in body frame
        a_x0 = acce[0]
        a_y0 = acce[1]
        a_z0 = acce[2]
        
        # Get change in Euler angles based on gyro data + dt 
        du = gyro[0]*dt
        dv = gyro[1]*dt
        dw = gyro[2]*dt
        
        u = u_old + du
        v = v_old + dv
        w = w_old + dw
        
        x_k[0] = x_0 + v_x0*dt + (c(w)*c(v)*a_x0 + ( c(w)*s(v)*s(u) - s(w)*c(u) )*a_y0 + ( c(w)*s(v)*c(u) + s(w)*s(u) )*a_z0) * dt**2 / 2    # x = x_0 + v_x*dt + a_x*dt^2 / 2
        x_k[1] = (c(dw)*c(dv)*v_x0 + ( c(dw)*s(dv)*s(du) - s(dw)*c(du) )*v_y0 + ( c(dw)*s(dv)*c(du) + s(dw)*s(du) )*v_z0) + (c(w)*c(v)*a_x0 + ( c(w)*s(v)*s(u) - s(w)*c(u) )*a_y0 + ( c(w)*s(v)*c(u) + s(w)*s(u) )*a_z0) * dt
        
        x_k[2] = y_0 + v_y0*dt + (s(w)*c(v)*a_x0 + ( s(w)*s(v)*s(u) + c(w)*c(u) )*a_y0 + ( s(w)*s(v)*c(u) - c(w)*s(u) )*a_z0) * dt**2 / 2
        x_k[3] = (s(dw)*c(dv)*v_x0 + ( s(dw)*s(dv)*s(du) + c(dw)*c(du) )*v_y0 + ( s(dw)*s(dv)*c(du) - c(dw)*s(du) )*v_z0) + (s(w)*c(v)*a_x0 + ( s(w)*s(v)*s(u) + c(w)*c(u) )*a_y0 + ( s(w)*s(v)*c(u) - c(w)*s(u) )*a_z0) * dt
        
        x_k[4] = z_0 + v_z0*dt + (-s(v)*a_x0 + ( c(v)*s(u) )*a_y0 + ( c(v)*c(u) )*a_z0) * dt**2 / 2
        x_k[5] = (-s(dv)*v_x0 + ( c(dv)*s(du) )*v_y0 + ( c(dv)*c(du) )*v_z0) + (-s(v)*a_x0 + ( c(v)*s(u) )*a_y0 + ( c(v)*c(u) )*a_z0 - 9.8) * dt
        
        x_k[6] = u
        x_k[7] = v 
        x_k[8] = w
        
        return x_k
        
    def get_H_f(self, acce, gyro, dt):
        ''' Get the Jacobian approximating the transformation function J_f
        '''
        
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        u = self.x_hatminus[6]
        v = self.x_hatminus[7]
        w = self.x_hatminus[8]
        
        du = gyro[0]*dt
        dv = gyro[1]*dt
        dw = gyro[2]*dt
        
        ax_s = acce[0]
        ay_s = acce[1]
        az_s = acce[2]
        
        # Partial deritaves
        xk_x = 1
        xk_vx = dt
        xk_y = 0
        xk_vy = 0
        xk_z = 0
        xk_vz = 0
        xk_u = (dt**2)*(ay_s*(s(v)* c(w) * c(u) + s(w)*s(u)) + az_s*( s(w)*c(u) - s(v)*c(w)*s(u)))/2
        xk_v = (dt**2)*(az_s*c(u)*c(v)*c(w) + ay_s*c(v)*c(w)*s(u) - ax_s*c(w)*s(v))/2
        xk_w = (dt**2)*(-ax_s*c(v)*s(w) + ay_s*( -c(w)*c(u) - s(u)*s(v)*s(w) ) + az_s*( c(w)*s(u) - c(u)*s(v)*s(w)))/2
        
        vxk_x = 0
        vxk_vx = c(dv)*c(dw)
        vxk_y = 0
        vxk_vy = c(dw)*s(du)*s(dv) - c(du)*s(dw)
        vxk_z = 0
        vxk_vz = c(du)*c(dw)*s(dv) + s(du)*s(dw)
        vxk_u = dt * ( ay_s*( c(u)*c(w)*s(v) + s(u)*s(w)) + az_s*( -c(w)*s(u)*s(v) + c(u)*s(w)) )
        vxk_v = dt * ( -ax_s*c(w)*s(v) + ay_s*s(u)*c(v)*c(w) + az_s*c(u)*c(v)*c(w) )
        vxk_w = dt * ( -ax_s*s(w)*c(v) + ay_s*(-c(u)*c(w) - s(u)*s(v)*s(w)) + az_s*( c(w)*s(u) - c(u)*s(v)*s(w)) )
        
        yk_x = 0
        yk_vx = 0
        yk_y = 1
        yk_vy = dt
        yk_z = 0
        yk_vz = 0
        yk_u = (dt**2)*( ay_s*( -c(w)*s(u) + c(u)*s(v)*s(w)) + az_s*( -c(u)*c(w) - s(u)*s(v)*s(w)) )/2
        yk_v = (dt**2)*( -ax_s*s(v)*s(w) + ay_s*s(u)*c(v)*s(w) + az_s*(c(u)*c(v)*s(w)) )/2
        yk_w = (dt**2)*( ax_s*c(v)*c(w) + ay_s*( s(u)*c(w)*s(v) - c(u)*s(w) ) + az_s*( c(u)*s(v)*c(w) + s(u)*s(w)))/2
        
        vyk_x = 0
        vyk_vx = c(dv)*s(dw)
        vyk_y = 0
        vyk_vy = c(du)*c(dw) + s(du)*s(dv)*s(dw)
        vyk_z = 0
        vyk_vz = -c(dw)*s(du) + c(du)*s(dv)*s(dw)
        vyk_u = dt * ( ay_s * (-s(u)*c(w) + c(u)*s(v)*s(w) ) + az_s * ( -c(u)*c(w) - s(u)*s(v)*s(w) ))
        vyk_v = dt * ( -ax_s * s(v) * s(w) + ay_s*s(u)*c(v)*s(w) + az_s*c(u)*c(v)*s(w) )
        vyk_w = dt * ( ax_s * c(v) * c(w) + ay_s*( s(u)*s(v)*c(w) - c(u)*s(w) ) + az_s*( c(u)*s(v)*c(w) + s(u)*s(w)) )
        
        zk_x = 0
        zk_vx = 0
        zk_y = 0
        zk_vy = 0
        zk_z = 1
        zk_vz = dt
        zk_u = (dt**2)*( ay_s*c(u)*c(v) - az_s*c(v)*s(u) )/2
        zk_v = (dt**2)*( -ax_s*c(v) - ay_s*s(u)*s(v) - az_s*c(u)*s(v) )/2
        zk_w = 0
        
        vzk_x = 0
        vzk_vx = -s(dv)
        vzk_y = 0
        vzk_vy = c(dv)*s(du)
        vzk_z = 0
        vzk_vz = c(du)*c(dv)
        vzk_u = dt*( ay_s*( c(u)*c(v) ) - az_s*( c(v)*s(u) ) )
        vzk_v = dt*( -ax_s*c(v) - ay_s*s(u)*s(v) - az_s*s(v)*c(u))
        vzk_w = 0
        
        J_f = np.array([[xk_x, xk_vx, xk_y, xk_vy, xk_z, xk_vz, xk_u, xk_v, xk_w],
                        [vxk_x, vxk_vx, vxk_y, vxk_vy, vxk_z, vxk_vz, vxk_u, vxk_v, vxk_w],
                        [yk_x, yk_vx, yk_y, yk_vy, yk_z, yk_vz, yk_u, yk_v, yk_w],
                        [vyk_x, vyk_vx, vyk_y, vyk_vy, vyk_z, vyk_vz, vyk_u, vyk_v, vyk_w],
                        [zk_x, zk_vx, zk_y, zk_vy, zk_z, zk_vz, zk_u, zk_v, zk_w],
                        [vzk_x, vzk_vx, vzk_y, vzk_vy, vzk_z, vzk_vz, vzk_u, vzk_v, vzk_w],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        return J_f
        
    def hfunc(self, state):
        ''' The observation function 
        
        z = h(x_)
        
        where z is the measurement vector and x_ is the state of the system vector
        '''
        
        z_meas = np.zeros(2)
        
        z = state[4]
        zdot = state[5]
        
        z_meas[0] = z
        z_meas[1] = zdot
        
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
        
        J = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0]])

        return J
        
    # Correct for gravity
    def correct_acce(self, acce):
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        g = 9.8
        
        u = -self.x_hatminus[6]
        v = -self.x_hatminus[7]
        w = -self.x_hatminus[8]
        
        ax_s = acce[0]
        ay_s = acce[1]
        az_s = acce[2]
        
        z_new = np.zeros(3)
        
        z_new[0] = c(w)*c(v)*ax_s + ( c(w)*s(v)*s(u) - s(w)*c(u) )*ay_s + ( c(w)*s(v)*c(u) + s(w)*s(u) )*az_s
        z_new[1] = s(w)*c(v)*ax_s + ( s(w)*s(v)*s(u) + c(w)*c(u) )*ay_s + ( s(w)*s(v)*c(u) - c(w)*s(u) )*az_s
        z_new[2] = -s(v)*ax_s + ( c(v)*s(u) )*ay_s + ( c(v)*c(u) )*az_s - g
        
        return z_new

    # Method to update step
    def update_state(self, acce, gyro, z, dt):
        ''' z in this case is acceleration data and gyro is (du/dt, dv/dt, and dw/dt), i.e. the readings from the
        gyroscope
        '''
        
        # Process noise variance
        self.Q = self.var_p * np.array([[dt**4/4, dt**3/2, 0, 0, 0, 0, dt**2/2, dt**2/2, dt**2/2],
                                        [dt**2/2, dt**2, 0, 0, 0, 0, dt, dt, dt],
                                        [0, 0, dt**4/4, dt**3/2, 0, 0, dt**2/2, dt**2/2, dt**2/2],
                                        [0, 0, dt**2/2, dt**2, 0, 0, dt, dt, dt],
                                        [0, 0, 0, 0, dt**4/4, dt**3/2, dt**2/2, dt**2/2, dt**2/2],
                                        [0, 0, 0, 0, dt**2/2, dt**2, dt, dt, dt],
                                        [0, 0, 0, 0, 0, 0, dt, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, dt, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, dt]])
        
        # Extrapolate next state
        self.x_hatminus = self.Ffunc(self.x_hat, acce, gyro, dt)
        
        # Get J_f matrix
        self.F = self.get_H_f(acce, gyro, dt)
        
        # Extrapolate uncertainty
        self.Pminus = self.F@self.P@np.transpose(self.F) + self.Q
        
        # Recalculate Jacobian
        self.J_h = self.get_H_j()
        
        # Calculate Kalman gain
        self.K = self.Pminus @ np.transpose(self.J_h) @ np.linalg.inv(self.J_h @ self.Pminus @ np.transpose(self.J_h) + self.R)
        
        # Correct estimation using measurement
        self.x_hat = self.x_hatminus + self.K@(np.array(z) - self.hfunc(self.x_hatminus))
        
        # Update estimate uncertainty
        self.P = (np.identity(9) - self.K@self.J_h)@self.Pminus
        
        self.sz += 1 
        
    # Return prediction
    def current_state(self):
        return self.x_hat
        
    # Initialize to generic state
    def initialize_state(self, x):
        self.x_hat = x
        
if (__name__ == "__main__"):
    
    ##################################
    #   Extended Test Case           #
    ##################################
    
    pf_filter = ExtendedKalman(1e-3, 2.0, 3.0, 10)
    pf_filter.initialize_state([0, 0, 0, 0, 0, 0, 0, 0, 0])
    x_estimate_states = []
    y_estimate_states = []
    z_estimate_states = []
    true_states= []
    times = []
    t = 0
    
    state = np.zeros(9)

    dt = 0.2
    for i in range(400):
        w_rate = 0.0
        
        acce = np.array([0.0, 0.0, 1.0]) #+ np.random.randn(3)/6.0
        acce = acce*9.8
        gyro = np.array([0, 0, w_rate])# +  (np.pi/180.0)*np.random.randn(3)/100.0
        
        #old_alt = altitude
        #altitude = (i*dt)**2*(np.sin(60.0*np.pi/180.0))*(5.0*9.8)/2
        
        #v_z = (altitude-old_alt)/dt
        z = np.array([0, 0])+ np.random.randn(2)/3.0
       
        pf_filter.update_state(acce, gyro, z, dt)
        state = pf_filter.current_state()
        
        print(f'Measured acce: {acce}')
        print(f'Measured gyro: {gyro}')
        print(f'Measured z + zdot {z}')
        print(state)
        x_estimate_states.append(state[0])
        y_estimate_states.append(state[2])
        z_estimate_states.append(state[4])
        
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