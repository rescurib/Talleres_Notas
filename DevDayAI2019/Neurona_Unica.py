# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:48:37 2019

@author: Rodolfo Escobar
"""

import numpy as np
import matplotlib.pyplot as plt

def logsig(x):
    """
    Función de activación
    """
    return 1/(1+np.exp(-x))

def df(z):
    """
    Derivada de la función de ativación
    """
    a = logsig(z)
    return a * (1.0 - a)

dataset = np.load("desgrad_dataset.npy")

p = dataset[:,0:2]
t = dataset[:,2]

def DG(p,t,Ep=10):
    """
    Entrenamiento por Descenso de Gradiente
    """
    # m será igual al número patrones de 
    # entrenamiento (ejemplos) y n al número
    # de elementos del vector de caracteristicas.
    m,n = p.shape
    a = 0.5
    
    #--- Pesos iniciales ---
    w = np.random.uniform(-0.25,0.25,2)
    b = np.random.uniform(-0.25,0.25)
    # ----------------------
    
    for N in range(Ep): # Iteración sobre num. de épocas
        for ti in range(m): # Iteración sobre num. de patrones 
            #---- Salida ----
            net = np.dot(w,p[ti])+b
            y = logsig(net)
            #-----------------
            #---Regla Delta---
            err = t[ti]- y             
            Delta = 2*err*df(net)*p[ti]
            w = w + a*Delta 
            b = b + a*2*err*df(net)
            #-----------------
             
    return w,b
    
#--- Entrenamiento --- 
w,b = DG(p,t,100)

#%%---- Gráficas del mapa de decisión ---

def modelo(X,w,b):
    z = np.dot(w,X)+b
    return logsig(z)

NN = 20 # Num de elementos de los intervalos de prueba
x0 = np.linspace(p.min()-0.5,p.max()+0.5,NN)
x1 = np.linspace(p.min()-0.5,p.max()+0.5,NN)
Zp = np.zeros([NN,NN]) # Espacio de prueba

plt.figure(1)
for i in range(NN):
    for j in range(NN):            #-----x-----#
        Zp[i][j] = np.round(modelo([x0[j],x1[i]],w,b))

X, Y = np.meshgrid(x0, x1)        
plt.contourf(X,Y,Zp,cmap = 'bwr',alpha=0.6)
plt.grid(True)
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")        
#-----------------------------------------
#--- Gráfica de datos de entrenamiento ---
for i in range(len(t)):
    if(t[i]==1):
        plt.scatter(p[i,0],p[i,1],c='r')
    else:
        plt.scatter(p[i,0],p[i,1],c='b')
#-----------------------------------------