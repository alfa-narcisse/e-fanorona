"""
- load all data as a npy file and stock them in the Directory datasets
"""

import numpy as np
X = np.random.randint(-2,2,(9,3))
Y = np.random.randn(16,3)
def initialisations(dimensions:list):
    c= len(dimensions)
    parameters = dict()
    for i in range(1,c):
        parameters["W"+str(i)] = np.random.randn(dimensions[i],dimensions[i-1])
        parameters["b"+str(i)] = np.random.randn(dimensions[i],1)
    return parameters

parameters = initialisations([9,9,32,64,16])

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
        max_col_j = np.max(Z[:,j])
        summ = np.sum(np.exp(Z[:,j] - max_col_j ))
        A[:,j] = np.exp(Z[:,j]-max_col_j)/summ
    return A


def forward_propagations(X, parameters):
    activations = {"A0": X}
    post_activations = {}
    C = len(parameters)//2
    for i in range(1,C):
        Z = parameters["W"+str(i)].dot(activations["A"+str(i-1)]) + parameters["b"+str(i)]
        activations["A"+str(i)] = ReLu(Z)
        post_activations["Z"+str(i)] = Z
    Z = parameters["W"+str(C)].dot(activations["A" + str(C - 1)]) + parameters["b" + str(C)]
    activations["A"+str(C)] = soft_max(Z)
    return activations, post_activations

activations, post_activations = forward_propagations(X, parameters)
#print(act["A4"])

# --------------- back propagation ---------------------

def back_propagation(parameters,post_activations, activations, Y):
    m = Y.shape[1]
    Im = np.ones((m,1))
    gradients = {}
    C = len(parameters)//2
    dZ = activations["A" +str(C)] - Y
    gradients["dW"+str(C)] = dZ.dot(activations["A"+str(C-1)].T)/m
    gradients["db"+str(C)] = dZ.dot(Im)/m

    for i in range(C-1,0,-1):
        dA = parameters["W"+str(i+1)].T.dot(dZ)
        maskI = post_activations["Z"+str(i)]>0
        dZ = dA*maskI
        gradients["dW"+str(i)] = dZ.dot(activations["A"+str(i-1)].T)/m
        gradients["db"+str(i)] = dZ.dot(Im)/m
    return gradients

"""gradients = back_propagation(parameters,post_activations, activations, Y)

for k,v in gradients.items():
    print(k, ": ", v.shape)"""

def update(gradients,parameters, lr):
    C = len(parameters)//2
    for i in range(1,C+1):
        parameters["W"+str(i)] -= lr*gradients["dW"+str(i)]
        parameters["b"+str(i)] -= lr*gradients["db"+str(i)]
    return parameters