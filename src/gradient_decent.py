from simulator import linear_params,easy_params,medium_params,hard_params
from generator import FDV
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import buildMatrices

def loss(states,predicted,estimAccel):
    return (np.sum((states@predicted-estimAccel)**2))/len(states)

#Testing data
#SNR = [100,10,5,2,1]
#for i in SNR:
#    data = np.load(f"C:\VS-Code\duffing-parameter-recovery\data\dataset_SNR{i}.npz")
#    clean = data["CleanStates"]
#    X, y = buildMatrices(clean[:, 0], clean[:, 1], data["timestep"])
#    pred = np.array([-1, -0.2])
#    print(f"SNR {i}: loss at true pred = {loss(X, pred, y):.2e}")

def grad(states,predicted,estimAccel):
    return (states.T@(states@predicted-estimAccel))/(len(states)/2)

def gradient_descent(params,epochs,lr):
    #-------------graphing
    lossCurve =[]
    alphaCurve = []
    gammaCurve = []
    #-------------
    pred = np.random.randn(2,)*np.sqrt(2)
    SNR = [100,10,5,2,1]
    for i in SNR:
        #data = np.load(f"C:\VS-Code Main\duffing-parameter-recovery\data\dataset_SNR{i}.npz") #dekstop
        data = np.load(f"C:\VS-Code\duffing-parameter-recovery\data\dataset_SNR{i}.npz") #laptop
        states,estimAccel = buildMatrices(data["NoisyDis"],data["NoisyVel"],data["timestep"])
        #normalisation
        mean = states.mean(axis=0)
        std = states.std(axis=0)
        states_normal = (states-mean)/std
        for epoch in range(epochs): 
            if (epoch+1) %100 ==0:   
                lr = lr*0.95
            if (epoch+1) %10 ==0:
                lossCurve.append(loss(states,pred,estimAccel))
            pred = pred-lr*grad(states_normal,pred,estimAccel)
            alphaCurve.append(pred[0]/std[0])
            gammaCurve.append(pred[1]/std[1])
        print(f"The loss for SNR of {i} is {loss(states,pred/std,estimAccel)} with the values of {(pred/std)*-1}")
        lr = lr
        pred = np.random.randn(2,)*np.sqrt(2)
        #graphing-------------
        plt.plot(np.arange(epochs,step=epochs/len(lossCurve)),lossCurve) #debug
        plt.show()
        fig,axs = plt.subplots(2)
        axs[0].plot(np.arange(epochs),-1*np.array(alphaCurve))
        axs[0].set_title("Alpha")
        axs[1].plot(np.arange(epochs),-1*np.array(gammaCurve))
        axs[1].set_title("Gamma")
        plt.show()
        alphaCurve = []
        gammaCurve = []
        lossCurve = []

gradient_descent(linear_params,100,0.1)

