"""
- load all data as a npy file and stock them in the Directory datasets
  



"""

import numpy as np
X = np.random.randint(-2,2,(9,3))
def initialisations(dimensions:list):
    c= len(dimensions)
    parameters = dict()
    for i in range(1,c):
        parameters["W"+str(i)] = np.random.randn(dimensions[i],dimensions[i-1])
        parameters["b"+str(i)] = np.random.randn(dimensions[i],1)
    return parameters

param = initialisations([9,9,32,64,16])
for k,v in param.items():
    print(k, v.shape)

def ReLu(Z):
    l,c = Z.shape
    A = np.zeros((l,c))
    for i in range(l):
        for j in range(c):
            A[i,j] = Z[i,j] if Z[i,j] >=0 else 0
    return A

def soft_max(Z):
    l,c = Z.shape
    A = np.zeros((l,c))
    for j in range(c):
        maxj = np.max(Z[:,j])
        summ = np.sum(np.exp(Z[:,j] - maxj))
        A[:,j] = np.exp(Z[:,j]-maxj)/summ
    return A


def forward_propagations(X, parameters):
    activations = {"A0": X}
    C = len(parameters)//2
    for i in range(1,C):
        Z = parameters["W"+str(i)].dot(activations["A"+str(i-1)]) + parameters["b"+str(i)]
        activations["A"+str(i)] = ReLu(Z)
    Z = parameters["W"+str(C)].dot(activations["A" + str(C - 1)]) + parameters["b" + str(C)]
    activations["A"+str(C)] = soft_max(Z)
    return activations

act = forward_propagations(X, param)
print(act["A4"])