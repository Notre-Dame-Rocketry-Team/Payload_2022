#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 17:30:32 2022
@author: dymatthews
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 07:26:30 2022
@author: dymatthews
"""

#grid_location_transformation
#2/6/22

import numpy as np
import math
import sys
 
def initialize_quadrants():
    x = 10 # number of squares in x direction
    y = 10 # number of squares in y direction
    a = np.zeros((x,y)) # positive x positive y
    b = np.zeros((x,y)) # negative x, positive y
    c = np.zeros((x,y)) # negative x, negative y
    d = np.zeros((x,y)) # positive x, negative y
    counter = 1
    '''initialize quadrant numbers with box 1 being at the extreme of the y-axis 
    and 100 at the extreme of the x-axis for that quadrant'''
    for i in range(x):
        for j in range(y):
            a[i][j] = counter
            b[i][9-j] = counter
            c[9-i][9-j] = counter
            d[9-i][j] = counter
            counter = counter+1
    return a,b,c,d

def create_quad_to_grid_arrays():
    x = 10 # number of squares in x direction
    y = 10 # number of squares in y direction
    n = x*y # number of total squares
    '''initialize a 100 unit long empty horizontal array for each quadrant 
    to represent the actual space of the quadrant location on the entire grid'''
    
    quad1 = np.zeros((1,n))# positive x positive y
    quad2 = np.zeros((1,n))# negative x, positive y
    quad3 = np.zeros((1,n))# negative x, negative y
    quad4 = np.zeros((1,n))# positive x, negative y
    
    q1_counter = 0
    q2_counter = 0
    q3_counter = 0
    q4_counter = 0
    
    ''' assign the actual numbered location on the full grid to 
    each quadrant array'''
    for i in range(1,401):
        if (i<=200 and (math.ceil(i/10)*10)%20!=0):
            quad2[0][q2_counter] = i
            q2_counter = q2_counter + 1
        elif (i<=200 and (math.ceil(i/10)*10)%20==0):
            quad1[0][q1_counter] = i
            q1_counter = q1_counter + 1
        elif (i>200 and (math.ceil(i/10)*10)%20!=0):
            quad3[0][q3_counter] = i
            q3_counter = q3_counter + 1
        elif (i>200 and (math.ceil(i/10)*10)%20==0):
            quad4[0][q4_counter] = i
            q4_counter = q4_counter + 1
    
    return quad1,quad2,quad3,quad4

def findPosition(inp_dx,inp_dy,quad_let,quad_num):
     if(inp_dx>2500 or inp_dy>2500):
         sys.exit("ERROR: Position out of bounds.")
         
     #determine x position
     dx_temp = abs(inp_dx)
     x_counter=0
     while dx_temp>0:
         dx_temp=dx_temp-250
         x_counter=x_counter+1
     dy_temp = abs(inp_dy)
     y_counter = 0
     #determine y position
     while dy_temp>0:
         dy_temp=dy_temp-250
         y_counter=y_counter+1
     
     '''If position value is 0, i.e. the rocket has landed directly 
     on the x or y axis then the position is defaulted to the grid 
     square on the positive side of the axis'''
     if(inp_dx == 0):
         x_counter =1
     if (inp_dy == 0):
         y_counter=1
     #print out quadrant position and final grid position
     quad_position = quad_let[10-(y_counter)][(x_counter-1)]
     grid_position = quad_num[0][int(quad_position-1)]
     print("quadrant position: " + str(quad_position))
     print(f'Grid position is: {grid_position}')
     return(grid_position)


def get_position(x_input, y_input):
    dx=x_input
    dy=y_input
       
    a,b,c,d = initialize_quadrants()
    quad1,quad2,quad3,quad4= create_quad_to_grid_arrays()
    
    if dx>=0 and dy>=0:
        print("Quadrant 1")
        findPosition(dx,dy,a,quad1)
    elif dx<0 and dy>=0:
        print("Quadrant 2")
        findPosition(dx,dy,b,quad2)   
    elif dx<0 and dy<0:
        print("Quadrant 3")
        findPosition(dx,dy,c,quad3)
    elif dx>=0 and dy<0:
        print("Quadrant 4")
        findPosition(dx,dy,d,quad4)