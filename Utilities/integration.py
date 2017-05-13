# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:40:22 2017

@author: ray
"""

def trip_trap_rule(a, hx, hy, hz):
    '''
    integration using trapzoidal rule
    a is the 3D np array which we want to integrate
    hx, hy, hz are the stepsizes at each dimension
    '''
    nx,ny,nz = a.shape
    count = 8    
    
    result = 0.
    
    
    #add each indices
    result += a[0][0][0] + a[0][0][nz-1] + a[0][ny-1][0] + a[0][ny-1][nz-1] +\
              a[nx-1][0][0] + a[nx-1][0][nz-1] + a[nx-1][ny-1][0] + a[nx-1][ny-1][nz-1]
    
    
    #add each edge
    temp = 0.
    for i in range(1,nz-1):
        temp += a[0][0][i] + a[0][ny-1][i] + a[nx-1][0][i] + a[nx-1][ny-1][i]
        count += 4
    for j in range(1,ny-1):
        temp += a[0][j][0] + a[0][j][nz-1] + a[nx-1][j][0] + a[nx-1][j][nz-1]
        count += 4
    for k in range(1,nx-1):
        temp += a[k][0][0] + a[k][0][nz-1] + a[k][ny-1][0] + a[k][ny-1][nz-1]
        count += 4
    
    result += temp*2.0
    
    
    #add each surface
    temp = 0.
    for j in range(1,ny-1):
        for i in range(1,nz-1):
            temp += a[0][j][i] + a[nx-1][j][i]
            count += 2
    for k in range(1,nx-1):
        for i in range(1,nz-1):
            temp += a[k][0][i] + a[k][ny-1][i]
            count += 2
    for k in range(1,nx-1):
        for j in range(1,ny-1):
            temp += a[k][j][0] + a[k][j][nz-1]
            count += 2
    result += temp*4.0
    
    
    #add points in the body
    temp = 0.
    for k in range(1,nx-1):
        for j in range(1,ny-1):
            for i in range(1,nz-1):
                temp += a[k][j][i]
                count += 1
    result += temp*8.0
    
#    print '\n\ncount'
#    print count
#    print '\n\n'
#    print result
    result *= ((hx*hy*hz)/8.0)
    return result
    


def trip_riemann_sum(a, hx, hy, hz):
    '''
    integration using Riemann sum
    a is the 3D np array which we want to integrate
    hx, hy, hz are the stepsizes at each dimension
    '''
    nx,ny,nz = a.shape
    count = 8    
    
    result = 0.
    
    
    #add each indices
    result += a[0][0][0] + a[0][0][nz-1] + a[0][ny-1][0] + a[0][ny-1][nz-1] +\
              a[nx-1][0][0] + a[nx-1][0][nz-1] + a[nx-1][ny-1][0] + a[nx-1][ny-1][nz-1]
    
    
    #add each edge
    temp = 0.
    for i in range(1,nz-1):
        temp += a[0][0][i] + a[0][ny-1][i] + a[nx-1][0][i] + a[nx-1][ny-1][i]
        count += 4
    for j in range(1,ny-1):
        temp += a[0][j][0] + a[0][j][nz-1] + a[nx-1][j][0] + a[nx-1][j][nz-1]
        count += 4
    for k in range(1,nx-1):
        temp += a[k][0][0] + a[k][0][nz-1] + a[k][ny-1][0] + a[k][ny-1][nz-1]
        count += 4
    
    result += temp#*2.0
    
    temp = 0.
    for j in range(1,ny-1):
        for i in range(1,nz-1):
            temp += a[0][j][i] + a[nx-1][j][i]
            count += 2
    for k in range(1,nx-1):
        for i in range(1,nz-1):
            temp += a[k][0][i] + a[k][ny-1][i]
            count += 2
    for k in range(1,nx-1):
        for j in range(1,ny-1):
            temp += a[k][j][0] + a[k][j][nz-1]
            count += 2
    result += temp#*4.0
    
    temp = 0.
    for k in range(1,nx-1):
        for j in range(1,ny-1):
            for i in range(1,nz-1):
                temp += a[k][j][i]
                count += 1
    result += temp#*8.0
    
#    print '\n\ncount'
#    print count
#    print '\n\n'
#    print result
    result *= ((hx*hy*hz))#/8.0)
    return result