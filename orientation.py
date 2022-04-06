''' Module containing objects and functions for manipulating values representing the vehicle's orientation. 

    by Juwan Jeremy Jacobe
    NDRT 2022
    
    Below are notes for the notation used for representing orientation values.
    
    Quaternions:
    Stored as a 4 dimensional np array
    q = [q_0, q_i, q_j, q_k]
    which represents the quaternion
    q = q_0 + i*q_i + j*q_j + k*q_k
    where i, j, and k are mutually orthogonal imaginary unit vectors. In our case, i, j, k can represent the x, y, z
    axes. 
    
    Euler angles:
    Stored as a three dimensional np array
    angle = [roll, pitch, yaw]
    where roll, pitch, yaw are the Euler angles, i.e. roll is angle of rotation with respect to x-axis/longtiudinal, 
    pitch is angle of rotation with respect to y-axis/lateral, and yaw is angle of rotation with respect to z-axis/vertical
'''
import numpy as np
from ahrs.filters import SAAM

def quaternion_mult(r, s):
    ''' Function to perform quaternion multiplication
    
    Args:
        r (array): quaternion 1
        s (array): quaternion 2
        
    Out:
        t (array): product of quaternion 1 and 2
    '''
    
    # Hold in intermediate values
    r0 = r[0]
    r1 = r[1]
    r2 = r[2]
    r3 = r[3]
    
    s0 = s[0]
    s1 = s[1]
    s2 = s[2]
    s3 = s[3]
    
    # Find components of product quaternion
    t0 = r0*s0 - r1*s1 - r2*s2 - r3*s3
    t1 = r0*s1 + r1*s0 - r2*s3 + r3*s2
    t2 = r0*s2 + r1*s3 + r2*s0 - r3*s1
    t3 = r0*s3 - r1*s2 + r2*s1 + r3*s0
    
    # Get product quaternion
    t = np.array([t0, t1, t2, t3])
    
    return t
    
def quaternion_inv(q):
    ''' Function to find the inverse of a quaternion
    
    Args:
        q (array): input quaternion
        
    Out:
        q_inv (array): inverse of input quaternion
    '''
    
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    
    return q_inv
    
class Orientation():
    ''' Class to describe the orientation of an object. Contains different representations of orientation, including
        Euler angles, quaternions, and rotation matrices
    '''
    
    def __init__(self, init_angle=np.zeros(3)):
        ''' Initialize orientation of vehicle, where angle is a vector containing the Euler angles. Angles are stored
        as radians instead of degrees!
        '''
        
        self.angle = self.Deg2Rad(init_angle)
        self.quat = self.Euler2Quaternion(init_angle)
    
    def Deg2Rad(self, degrees):
        return degrees*np.pi/180.0
        
    def Rad2Deg(self, rads):
        return rads*180.0/np.pi
    
    def Euler2Quaternion(self, angle):
        ''' Method to transform Euler angles to quaternions
        '''
        
        # Get roll pitch yaw from angle
        roll = angle[0]
        pitch = angle[1]
        yaw = angle[2]
        
        # Reexpress cos and sin as c and s for readibility
        c = lambda x: np.cos(x)
        s = lambda x: np.sin(x)
        
        # Calculate quaternion components
        q0 = c(roll/2) * c(pitch/2) * c(yaw/2) + s(roll/2) * s(pitch/2) * s(yaw/2)
        qi = s(roll/2) * c(pitch/2) * c(yaw/2) - c(roll/2) * s(pitch/2) * s(yaw/2)
        qj = c(roll/2) * s(pitch/2) * c(yaw/2) + s(roll/2) * c(pitch/2) * s(yaw/2)
        qk = c(roll/2) * c(pitch/2) * s(yaw/2) - s(roll/2) * s(pitch/2) * c(yaw/2)
        
        return np.array([q0, qi, qj, qk])
    
    def Quaternion2Angle(self, q):
        ''' Transform quaternion to Euler angle
        '''
        # Intermediate values
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        
        tolerance = 0.00001
        
        # Calculate pitch first to check gimbal lock condition
        pitch = np.arcsin(2*(q0*q2 - q1*q3))
        
        # First two cases to prevent gimbal lock. Otherwise, calculate as normal
        if abs(pitch - np.pi/2) < tolerance:
            roll = 0
            yaw = -2*np.arctan2(q1, q0)
        elif abs(pitch + np.pi/2) < tolerance:
            roll = 0
            yaw = 2*np.arctan2(q1, q0)
        else:
            roll = np.arctan2( 2*(q0*q1 + q2*q3), q0**2 - q1**2 - q2**2 + q3**2)
            yaw = np.arctan2( 2*(q0*q3 + q1*q2), q0**2 + q1**2 - q2**2 - q3**2)
        
        return np.array([roll, pitch, yaw])
        
    def update_angle(self, d_angle):
        ''' Update Euler angles, where d_angle is a 3d vector containing the change in the three Euler angles
        '''
        
        # Update Euler angle and quaternions
        self.angle += d_angle
        self.quat = self.Euler2Quaternion(self.angle)
        
    def update_quaternion(self, q_new):
        ''' Update quaternion and corresponding to that update the angle. This will used for when q_new is the "filtered"
        quaternion
        '''
        
        self.quat = q_new
        self.angle = self.Quaternion2Angle(q_new)
    
    def construct_rot_matrix(self):
        ''' Construct a rotation matrix based on current orientation, where
             _           _
            | r11 r12 r13 |
        R = | r21 r22 r23 |
            |_r31 r32 r33_|
             
        '''
        q0 = self.quat[0]
        q1 = self.quat[1]
        q2 = self.quat[2]
        q3 = self.quat[3]
        
        # Calculate rotation matrix components
        r11 = q0**2 + q1**2 - q2**2 - q3**2
        r12 = 2*q1*q2 - 2*q0*q3
        r13 = 2*q1*q3 + 2*q0*q2
        
        r21 = 2*q1*q2 + 2*q0*q3
        r22 = q0**2 - q1**2 + q2**2 - q3**2
        r23 = 2*q2*q3 - 2*q0*q1
        
        r31 = 2*q1*q3 - 2*q0*q2
        r32 = 2*q2*q3 + 2*q0*q1
        r33 = q0**2 - q1**2 - q2**2 + q3**2
        
        # Construct matrix from components
        self.R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        

if __name__ == '__main__':        
    num = 1
    acce = np.array([4.098297, 8.663757, 2.1355896])
    mag = np.array([-28.71550512, -25.92743566, 4.75683931])

    attitude_estimator = SAAM()
    attitude = attitude_estimator.estimate(acc=acce, mag=mag)

    orientation = Orientation()
    orientation.update_quaternion(attitude)

    print(attitude)
    print(orientation.Rad2Deg(orientation.angle))