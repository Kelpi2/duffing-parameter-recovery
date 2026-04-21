from simulator import linear_params,easy_params,medium_params,hard_params
from generator import FDV
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import buildMatrices

def loss(states,predicted,estimAccel):
    return (np.sum((states@predicted-estimAccel)**2))/len(states)
SNR = [100,10,5,2,1]
for i in SNR:
    data = np.load(f"C:\VS-Code\duffing-parameter-recovery\data\dataset_SNR{i}.npz")
    clean = data["CleanStates"]
    X, y = buildMatrices(clean[:, 0], clean[:, 1], data["timestep"])
    pred = np.array([-1, -0.2])
    print(f"SNR {i}: loss at true pred = {loss(X, pred, y):.2e}")

def grad(states,predicted,estimAccel):
    return (states.T@(states@predicted-estimAccel))/(len(states)/2)

def gradient_descent(params,epochs):
    #lossY =[]
    pred = np.random.randn(2,)*np.sqrt(2)
    lr = 0.001
    SNR = [100,10,5,2,1]
    for i in SNR:
        #data = np.load(f"C:\VS-Code Main\duffing-parameter-recovery\data\dataset_SNR{i}.npz") #dekstop
        data = np.load(f"C:\VS-Code\duffing-parameter-recovery\data\dataset_SNR{i}.npz") #laptop
        states,estimAccel = buildMatrices(data["NoisyDis"],data["NoisyVel"],data["timestep"])
        for epoch in range(epochs):
            if (epoch+1) %100 ==0:
                lr = lr*0.95
                #lossY.append(loss(states,pred,estimAccel))
            pred = pred-lr*grad(states,pred,estimAccel)
        print(f"The loss for SNR of {i} is {loss(states,pred,estimAccel)} with the values of {pred}")
        lr = 0.001
        pred = np.random.randn(2,)*np.sqrt(2)
        #plt.plot(np.arange(1000,step=100),lossY) #debug
        #plt.show()
        #lossY = []

#gradient_descent(linear_params,1000)

