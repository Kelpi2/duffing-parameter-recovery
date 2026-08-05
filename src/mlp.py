import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
from generator import MLPdataset,addNoise
np.random.seed(200)

#dataset stuff
def prepareData(num,generate,SNR):
    if generate:
        (X,Y) = MLPdataset(num)
    else:
        data = np.load(os.path.join(DATA_DIR, f"MLPdataset.npz"))
        (X,Y) = (data["trainSet"],data["TruthLabels"])
    if SNR != "clean":
        SD = X.std(axis=1, keepdims=True)/SNR
        X = addNoise(X,SD)
    
    Tnum = int(num*0.6)
    Vnum = int(num*0.2) + Tnum
    X_train = X[:Tnum]
    X_eval = X[Tnum:Vnum]
    X_test = X[Vnum:]
    Xmean = np.mean(X_train)
    Xstd = np.std(X_train)
    X_train = (X_train-Xmean)/Xstd
    X_test = (X_test-Xmean)/Xstd
    X_eval = (X_eval-Xmean)/Xstd

    Y_train = Y[:Tnum]
    Y_eval = Y[Tnum:Vnum]
    Y_test = Y[Vnum:]
    Ymean = np.mean(Y_train,axis = 0)
    Ystd = np.std(Y_train,axis = 0)
    Y_train = (Y_train-Ymean)/Ystd
    Y_eval = (Y_eval-Ymean)/Ystd

    return X_train,X_test,Y_train,Y_test,Ymean,Ystd,X_eval,Y_eval

#initialize weights & functions

def initWeights():
    sizes = [604,128,64,32,16,5] 
    weights,biases = [],[]
    for wIn, wOut in zip(sizes[:-1],sizes[1:]):
        weights.append(np.random.randn(wIn,wOut)*np.sqrt(1/wIn))
        biases.append(np.zeros((1,wOut)))
    return weights,biases
        
def Tanh(x):
    return np.tanh(x)

def DervTanh(x):
    return 1-np.tanh(x)**2

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

def trainLoop(X_train,Y_train,X_eval,Y_eval,epochs,plot,seed):
    np.random.seed(seed)
    lr = 0.01
    xaxis = np.arange(epochs)
    tempYaxis = yaxis = np.zeros(epochs)
    yaxis = np.zeros(epochs)
    weights,biases = initWeights()
    minloss = 0

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
        #Smallest loss
        raw,activated = forward(X_eval,weights,biases)
        temploss = loss(activated[-1],Y_eval)
        if temploss < minloss or minloss == 0:
            minloss = temploss
            minWeights,minBiases = weights.copy(),biases.copy()
            minEpoch = epoch
        tempYaxis[epoch] = temploss
        yaxis[epoch] = epochLoss/(counter)
        if epoch % 50 ==0 :
            print(epochLoss/counter)
    if plot:
        plt.plot(xaxis,yaxis,label = "Loss")
        plt.plot(xaxis,tempYaxis,label = "TempLoss")
        plt.legend()
        plt.show()
    print(minEpoch)    

    return minWeights,minBiases

def accuracy(activated,Y,Ymean,Ystd):
    pred = activated[-1]*Ystd+Ymean
    res = np.sum((pred-Y)**2,axis = 0)
    testMean = np.mean(Y, axis=0)
    tot = np.sum((Y - testMean)**2,axis=0)
    params = 1-res/tot
    print(params)
    print(np.sqrt(np.mean((pred - Y)**2, axis=0)))

def test(X,Y,weights,biases,Ymean,Ystd,test):
    if test:
        print("Test set")
        __,activated = forward(X,weights,biases)
        accuracy(activated,Y,Ymean,Ystd)

if __name__ == "__main__":
    X_train,X_test,Y_train,Y_test,Ymean,Ystd,X_eval,Y_eval = prepareData(5000,0,1) #Trajectories,gen new data,SNR
    weights,biases = trainLoop(X_train,Y_train,X_eval,Y_eval,1000,1,100)  #X,Y,X_val,Y_val,Epochs,Plots,Seed
    __,activated = forward(X_eval,weights,biases)

    print("Eval set")
    accuracy(activated,(Y_eval*Ystd+Ymean),Ymean,Ystd)
    test(X_test,Y_test,weights,biases,Ymean,Ystd,1)

    #fit check
    print("Fit check")
    __,activated = forward(X_train,weights,biases)
    accuracy(activated,(Y_train*Ystd+Ymean),Ymean,Ystd)
    