import numpy as np
import matplotlib.pyplot as plt
import pickle
import timeit
import random
import pandas as pd

#Activation function
def functionSigma(x,out,c):
    s = np.zeros((out,1))
    nin = np.shape(x)[0]
    
    for neuronsOut in range(out):
        z = 0
        for neuronsIn in range(nin):
            z += x[neuronsIn]*c[neuronsIn,neuronsOut]
        s[neuronsOut,0] = ((1/(1 + np.exp(z))))
    return s

def functionPhi(x,out,uSize,c):
    s = np.zeros((out,1))#Variable para guardar resultados
    nin = np.shape(x)[0]
    
    for neuronsOut in range(out):
        z = 0
        for neuronsIn in range(nin):
            z += x[neuronsIn]*c[neuronsIn,neuronsOut]
        s[neuronsOut,0] = ((1/(1 + np.exp(z))))*(1)
    s = s@np.ones((1,uSize))#Multiplicación por una matriz
    return s

def DSigma(x,out,c):
    nin = np.shape(x)[0]
    s = np.zeros((out,nin))
    Sigma = functionSigma(x,out,c)
    for neuronsIn in range(nin):
        for neuronsOut in range(out):
            Sigma_i = Sigma[neuronsOut,0]
            C_i = c[neuronsIn,neuronsOut]
            s[neuronsOut,neuronsIn]=(Sigma_i*(1-Sigma_i)*C_i)
    return s

def DPhi(x,out,u,c):
    nin = np.shape(x)[0]
    s = np.zeros((out,nin))
    uSize = np.shape(u)[0]
    Phi = functionPhi(x,out,uSize,c)
    for u_i in range(uSize):
        for neuronsIn in range(nin):
            for neuronsOut in range(out):
                Phi_i = Phi[neuronsOut,u_i]
                C_i = c[neuronsIn,neuronsOut]
                s[neuronsOut,neuronsIn] += (Phi_i*(1-Phi_i)*C_i)*u[u_i,0]
    return s


#My previous learning laws were to dependent on other functions. In this version,there are independent


def LAW1(W_a,delta,g,P,A,Psi,partial=10):
    nRow, nCol = W_a.shape
    G = (P@A@delta@Psi.T)/2
    P = 7*P
    Psi=Psi/(2**(1/2))
    gLeft = np.identity(nRow)*g
    gRight = np.identity(nCol)*g
    leftT = np.linalg.inv(gLeft + P@P)
    rightT = np.linalg.inv(gRight + (Psi@Psi.T)@(Psi@Psi.T))
    t1 = -g*g*W_a
    t2 = -g*G
    t3 = -g**(1/2)*partial*G@(Psi@Psi.T)
    t4 = -g**(1/2)*partial*P@G
    t5 = P@P@W_a@(Psi@Psi.T)@(Psi@Psi.T)
    t6 = -P@G@(Psi@Psi.T)
    W = -leftT@(t1+t2+t3+t4+t5+t6)@rightT
    return W
#Second Layer or internal Layer
def LAW2(W21_a,W1s,delta,k,P,xh,A,n,Derivative):
    W21_aF = np.reshape(np.matrix.flatten(W21_a),(-1,1))#Vectorize the weights vector and also reshape the vector
    X_M = np.kron(np.identity(n),xh.T) #Create Kronecker product
    Gamma=W1s@Derivative@X_M #Compute Gamma
    stateSize = np.shape(xh)[0] #Size of the wieghts
    I_K = np.identity(np.shape(W21_a)[0]*np.shape(W21_a)[1])*k #Create identity matrix
    Inverse = np.linalg.inv(I_K + 14*Gamma.T@P@Gamma/1) # Compute inverse of matrix
    W1 = (W21_aF.T@(I_K - 14*Gamma.T@P@Gamma/1)@Inverse + 1*delta.T@A.T@P@Gamma@Inverse) #Update W
    return np.reshape(W1,(-1,stateSize))
#Learning Laws Hidden Layer
def LAW3(W21_a,W1s,W21s,delta,k,P,xh,A,n,Derivative1,Derivative2):
    W21_aF = np.reshape(np.matrix.flatten(W21_a),(-1,1))#Vectorize the weights vector and also reshape the vector
    X_M = np.kron(np.identity(n),xh.T) #Create Kronecker product
    Gamma=W1s@Derivative1@W21s@Derivative2@X_M #Compute Gamma
    stateSize = np.shape(xh)[0] #Size of the wieghts
    I_K = np.identity(np.shape(W21_a)[0]*np.shape(W21_a)[1])*k #Create identity matrix
    Inverse = np.linalg.inv(I_K + 56*Gamma.T@P@Gamma/1) # Compute inverse of matrix
    W1 = (W21_aF.T@(I_K - 56*Gamma.T@P@Gamma/1)@Inverse + 1*delta.T@A.T@P@Gamma@Inverse) #Update W
    return np.reshape(W1,(-1,stateSize))