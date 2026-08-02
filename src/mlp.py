import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from generator import MLPdataset

#dataset stuff
def prepareData(num):
    (X,Y) = MLPdataset(num)
    num = int(num*0.8)
    X_train = X[:num]
    X_test = X[num:]
    Xmean = np.mean(X_train)
    Xstd = np.std(X_train)
    X_train = (X_train-Xmean)/Xstd
    X_test = (X_test-Xmean)/Xstd

    Y_train = Y[:num]
    Y_test = Y[num:]
    Ymean = np.mean(Y_train,axis = 0)
    Ystd = np.std(Y_train,axis = 0)
    Y_train = (Y_train-Ymean)/Ystd

    return X_train,X_test,Y_train,Y_test,Ymean,Ystd

#initialize weights & functions

def initWeights():
    sizes = [604,128,5] 
    weights,biases = [],[]
    for wIn, wOut in zip(sizes[:-1],sizes[1:]):
        weights.append(np.random.randn(wIn,wOut)*np.sqrt(1/wIn))
        biases.append(np.zeros((1,wOut)))
    return weights,biases
        
def Tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def DervTanh(x):
    return 1-Tanh(x)**2

def loss(pred, y):
    return np.mean((pred - y)**2)

def forward(X,weights,biases):
    raw = []
    activated = [X]
    for i in range(len(weights)):
        raw.append(activated[i]@weights[i]+biases[i])
        if i == len(weights)-1:
            activated.append(raw[i])
        else:
            activated.append(Tanh(raw[i]))
    return raw,activated

#backprop

def backwards(y,raw,act,weights):
    act = act[::-1]
    raw = raw[::-1]
    weights = weights[::-1]
    dervW = []
    dervRaw = []
    dervAct = []
    dervB = []
    for i in range(len(weights)):
        if i == 0:
            dervRaw.append(act[i]-y)
            dervRaw[0] = dervRaw[0]/y.shape[0]
            dervW.append(act[i+1].T@dervRaw[i])
            dervB.append(np.sum(dervRaw[i],axis=0,keepdims=True))
            dervAct.append(dervRaw[i]@weights[i].T)
        else:
            dervRaw.append(dervAct[i-1]*DervTanh(raw[i]))
            dervW.append(act[i+1].T@dervRaw[i])
            dervB.append(np.sum(dervRaw[i],axis=0,keepdims=True))
            dervAct.append(dervRaw[i]@weights[i].T)
    return dervW[::-1],dervB[::-1]

#update

def update(weights, biases, dervW,dervB,lr):
    for i in range(len(weights)):
        weights[i] = weights[i] - lr*dervW[i]
        biases[i] = biases[i] - lr*dervB[i]
    return weights, biases

def trainLoop(X_train,Y_train,epochs):
    lr = 0.01
    xaxis = np.arange(epochs)
    yaxis = np.zeros(epochs)
    weights,biases = initWeights()

    for epoch in range(epochs):
        sIndex = np.random.permutation(len(X_train))
        X_train, Y_train = X_train[sIndex], Y_train[sIndex]
        epochLoss = 0
        counter = 0
        for index in range(0,len(X_train), 32):
            X = X_train[index:index+32]
            y = Y_train[index:index+32]
            raw,activated = forward(X,weights,biases)
            epochLoss += loss(activated[-1],y)
            dervW, dervB = backwards(y,raw,activated,weights)
            weights,biases = update(weights,biases,dervW,dervB,lr)
            counter += 1
        yaxis[epoch] = epochLoss/(counter)
        if epoch % 50 ==0 :
            print(epochLoss/counter)
    return weights,biases

def accuracy(Y_test,Ymean,Ystd):
    pred = activated[-1]*Ystd+Ymean
    res = np.sum((pred-Y_test)**2,axis = 0)
    testMean = np.mean(Y_test, axis=0)
    tot = np.sum((Y_test - testMean)**2,axis=0)
    params = 1-res/tot
    print(params)
    print(np.sqrt(np.mean((pred - Y_test)**2, axis=0)))

if __name__ == "__main__":
    X_train,X_test,Y_train,Y_test,Ymean,Ystd = prepareData(1000)
    weights,biases = trainLoop(X_train,Y_train,1000)
    __,activated = forward(X_test,weights,biases)
    accuracy(Y_test,Ymean,Ystd)
