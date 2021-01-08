# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:39:35 2018

@author: vishnu
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

width = 640
height = 480
img = np.ones((height,width))

plt.figure()
imgplot = plt.imshow(img)
plt.show()
#%%
X = ([np.array([304., 304.]),np.array([304., 305.]),np.array([304., 360.]),np.array([360., 376.]),np.array([305., 357.]),np.array([357., 375.]),\
             np.array([304., 379.]),np.array([379., 373.]),np.array([373., 402.]),np.array([304., 376.]),np.array([376., 363.]),np.array([363., 406.]),\
             np.array([304., 280.]),np.array([280., 273.]),np.array([273., 273.]),np.array([280., 274.]),np.array([274., 277.])])
Y = np.array([[170., 139.],[170., 200.],[139., 137.],[137., 171.],[200., 203.],[203., 189.],\
             [170., 155.],[155., 137.],[137., 196.],[170., 192.],[192., 241.],[241., 170.],\
             [170., 173.],[173., 167.],[167., 156.],[173., 180.],[180., 190.]])

Xmax = np.amax(X)
Ymax = np.amax(Y)
Xmin = np.amin(X)
Ymin = np.amin(Y)
print(Xmax,Xmin,Ymax,Ymin)
Xdiff = Xmax - Xmin
Ydiff = Ymax - Ymin
print(Xdiff,Ydiff)
X = X - Xmin + 1
Y = Y - Ymin + 1

if Xdiff > Ydiff:
    Smove = (Xdiff - Ydiff)/2
    Side = Xdiff
    Y = Y + Smove
else:
    Smove = (Ydiff - Xdiff)/2
    Side = Ydiff
    X = X + Smove
X = np.around(X)
Y = np.around(Y)

X = np.around(X * (19/Side)) + 1
Y = np.around(Y * (19/Side)) + 1
X = X.astype(int)
Y = Y.astype(int) 
print(X,Y)  
for i in range(17):
    plt.plot(X[i], Y[i], marker = 'o')
plt.show()

#%%
plt.plot([100,20], [15,35], marker = 'o')
plt.show()
X1 = np.array([[1,2],[3,4]])
print(X1.shape)
#%%

a = ([[1,2],[3,4]])
#a1 = np.append((a))
#print(a1)
#a2 = np.asarray(a1)
#print(a1)
print(np.shape(a))

temp = a
a = []
print(temp)
for l in temp:
    a.extend(l)
print(a)
#%%
import numpy as np
a = [1,2,3]
a1 = np.array(a)
print(a1[0])
#%%
















