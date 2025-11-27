"""
- load all data as a npy file and stock them in the Directory datasets
"""
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt



def initialisations(dimensions:list):
    c= len(dimensions)
    parameters = dict()
    for i in range(1,c):
        parameters["W"+str(i)] = np.random.randn(dimensions[i],dimensions[i-1])
        parameters["b"+str(i)] = np.random.randn(dimensions[i],1)
    return parameters

#parameters = initialisations([9,9,32,64,16])

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
def Cross_entropy(A, B, epsilon=1e-12):
    """
    Cross-entropy loss pour la classification multi-classe
    """
    A = np.clip(A, epsilon, 1.0 - epsilon)
    m = A.shape[1]
    return -np.sum(B * np.log(A)) / m


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

#activations, post_activations = forward_propagations(X, parameters)
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

def predict(X, parameters):
    activations,_ = forward_propagations(X,parameters)
    C = len(parameters)//2
    Af = activations["A"+str(C)]
    return (Af == Af.max(axis=0, keepdims = True)).astype(int)

def neural_network(X,Y, hidden_layer = (9,32,64), lr = 0.1, n_iter=1000):
    np.random.seed()
    dimensions = list(hidden_layer)
    dimensions.insert(0, X.shape[0])
    m = X.shape[1]
    dimensions.append(Y.shape[0])
    #parameters = initialisations(dimensions)
    param = np.load('fanorona_parameters.npz')
    parameters = {k:v for k,v in param.items()}
    train_loss = []
    train_acc = []
    for i in  tqdm(range(n_iter)):
        activations, post_activations = forward_propagations(X,parameters)
        gradients = back_propagation(parameters, post_activations, activations, Y)
        parameters = update(gradients,parameters, lr)

        if i%10 == 0:
            C = len(parameters)//2
            train_loss.append(Cross_entropy(activations["A"+str(C)], Y))
            Y_pred = predict(X,parameters)
            current_acc = accuracy_score( Y.flatten(), Y_pred.flatten())
            train_acc.append(current_acc)

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize = (18,4))
    ax[0].plot(train_loss, label="Train_loss")
    ax[0].legend()

    ax[1].plot(train_acc, label="Train_Accuracy")
    ax[1].legend()
    plt.show()
    return parameters, train_acc[-1], train_loss[-1]

def save_parameters(parameters):
    np.savez("fanorona_parameters_v1.npz", **parameters)

# loading data
dataset = np.load('datasets/datasets.npz')
X = dataset["X"]
Y = dataset["Y"]
parameters,t_a, t_l = neural_network(X,Y)
save_parameters(parameters)
res = {"res": np.array([t_a, t_l])}
np.savez('res_train.npz', **res)
print(t_a, t_l)
