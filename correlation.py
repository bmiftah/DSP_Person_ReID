from numpy import *
import numpy as np 
a = array([1,2,3,4,6,7,8,9])
b = array([2,4,6,8,10,12,13,15])
c = array([-1,-2,-2,-3,-4,-6,-7,-8])

a = np.random.random((256,32,96))
b = np.random.random((256,32,96))
print(" a ",np.shape(a))
a_channel = a[:][:][0]
b_channel = a[:][:][0]
a_flatten = a_channel.flatten()
b_flatten = b_channel.flatten()

print(np.shape(a_flatten))
print(np.shape(b_flatten))
corr = corrcoef(a_flatten,b_flatten,'valid')
print(" corr ",corr)