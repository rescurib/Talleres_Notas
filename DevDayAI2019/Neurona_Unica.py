# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:48:37 2019

@author: Rodolfo Escobar
"""

import numpy as np
import matplotlib.pyplot as plt

def logsig(x):
    return 1/(1+np.exp(-x))

def df(z):
    a = logsig(z)
    return a * (1.0 - a)

dataset = np.load("desgrad_dataset.npy")

p = dataset[:,0:2]
t = dataset[:,2]

def DG(p,t,Ep=10):
    """
    Entrenamiento por Decenso de Gradiente
    """
    m,n = p.shape
    a = 0.5
    
    #--- Pesos iniciales ---
    w = np.random.uniform(-0.25,0.25,2)
    b = np.random.uniform(-0.25,0.25)
    # ----------------------
    
    Delta = np.zeros([m,2])

    for N in range(Ep):
        for ti in range(m):
            #--- Salida
            net = np.dot(w,p[ti])+b
            y = logsig(net)
            #----

            err = t[ti]- y             
            Delta = 2*err*df(net)*p[ti]
            w = w + a*Delta
            b = b + a*2*err*df(net)
              
        w += a*Delta.mean(axis=0)
        b += a*Delta.mean(axis=0)
        
    return w,b

def modelo(X,w,b):
    z = np.dot(w,X)+b
    return logsig(z)
    
#--- Entrenamiento --- 
w,b = DG(p,t,100)

#%%---- Graficas ---
def hardlim(z,lim = 0):
    return 1 if z>lim else 0
#------------
NN = 20
x0 = np.linspace(p.min()-0.5,p.max()+0.5,NN)
x1 = np.linspace(p.min()-0.5,p.max()+0.5,NN)
Zp = np.zeros([NN,NN])

plt.figure(1)
for i in range(NN):
    for j in range(NN):             #-----x------#
        Zp[i][j] = hardlim(np.dot(w,[x0[j],x1[i]])+b)#hardlim(modelo([Xt[j],Yt[i]],w,b),0.5)

X, Y = np.meshgrid(x0, x1)        
plt.contourf(X,Y,Zp,cmap = 'bwr',alpha=0.6)
plt.grid(True)
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")        
#----------------------

for i in range(len(t)):
    if(t[i]==1):
        plt.scatter(p[i,0],p[i,1],c='r')
    else:
        plt.scatter(p[i,0],p[i,1],c='b')