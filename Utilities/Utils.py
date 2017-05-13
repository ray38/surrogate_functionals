import numpy as np

def neighbors(arr,position,n=3):
    ''' Given a 3D-array, and position=(x,y,z) returns an nxnxn array whose "center" element is arr[x,y,z]'''
    x,y,z = position
    arr=np.roll(np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1),shift=-z+1,axis=2)
    return arr[:n,:n,:n]
    
def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return
    
def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn))
    
def map_back(arr, maxx, minn):
    return np.add(np.multiply(arr,(maxx-minn)),minn)