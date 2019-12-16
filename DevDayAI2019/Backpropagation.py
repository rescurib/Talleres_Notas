# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 20:58:35 2019

@author: Rodolfo Escobar
"""

import numpy as np
import matplotlib.pyplot as plt

def logsig(x):
    return 1/(1+np.exp(-x))

def df(z):
    a = logsig(z)
    return a * (1.0 - a)

#dataset = np.load("dataset_BP_blobs.npy")
dataset = np.load("dataset_BP_moons.npy")

p = dataset[:,0:2]
t = dataset[:,2:4]

def BP(p,t,Ep=10):
    """
    Entrenamiento por Backpropagation
    """
    rate = 0.5 #Razon de aprendizaje 
    #--- Pesos y bias iniciales ---
    Neuronas_1_capa = 4
    Neuronas_2_capa = t.shape[1]
    W2 = np.random.uniform(-0.25,0.25,[Neuronas_1_capa,Neuronas_2_capa])
    W1 = np.random.uniform(-0.25,0.25,[p.shape[1],Neuronas_1_capa])
    b2 = np.random.uniform(-0.25,0.25)
    b1 = np.random.uniform(-0.25,0.25,[Neuronas_1_capa])
    # ----------------------
    for N in range(Ep):
        for k in range(p.shape[0]): # de 0 a Num. de ejemplos
            #--Operaciones de red
            net1 = np.dot(W1.T,p[k])+b1
            h = logsig(net1)
            net2 = np.dot(W2.T,h)+b2
            y = logsig(net2)
            e = t[k] - y
            #---
            #---Sensibilidades
            S2 = 2*e*df(net2)   #Sensibilidad Capa de salida
            S1 = df(net1)*np.dot(W2,S2) #Sensibilidad Capa de entrada
            #---
            #--- Regla delta generalizada
            for i in range(W2.shape[0]): #Ajuste de pesos de L2 (salida)
               for j in range(W2.shape[1]):
                   W2[i][j] += rate*S2[j]*h[i]
            
            b2 += rate*S2 #Ajuste de bias de L2       
            
            for i in range(W1.shape[0]): #Ajuste de pesos de L1 (capa oculta)
                for j in range(W1.shape[1]):
                   W1[i][j] += rate*S1[j]*p[k][i]
          
            b1 += rate*S1 #Ajuste de bias de entrada

    return W1,W2,b1,b2

#--- Entrenamiento --- 
W1,W2,b1,b2 = BP(p,t,1000)

#%%---- Graficas ---
def RedNeuronal(x,W1,W2,b1,b2):
    net1 = np.dot(W1.T,x)+b1
    h = logsig(net1)
    net2 = np.dot(W2.T,h)+b2
    y = logsig(net2)
    return y
##--------
NN = 20
Xt = np.linspace(p.min()-0.5,p.max()+0.5,NN)
Yt = np.linspace(p.min()-0.5,p.max()+0.5,NN)
Zp = np.zeros([NN,NN])

plt.figure(1)
for i in range(NN):
    for j in range(NN):
        Zp[i][j] = np.round(RedNeuronal([Xt[j],Yt[i]],W1,W2,b1,b2)[0])

X, Y = np.meshgrid(Xt, Yt)        
plt.contourf(X,Y,Zp,cmap = 'bwr',alpha=0.6)
plt.grid(True)
plt.xlabel("$x_0$")
plt.ylabel("$x_1$")          
##--------------------
for i in range(len(t)):
    if(t[i][0]==1):
        plt.scatter(p[i,0],p[i,1],c='r')
    else:
        plt.scatter(p[i,0],p[i,1],c='b')