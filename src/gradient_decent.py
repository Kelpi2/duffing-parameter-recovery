from simulator import linear_params,easy_params,medium_params,hard_params
from generator import FDV
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import buildMatrices
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def loss(states,predicted,estimAccel):
    return (np.sum((states@predicted-estimAccel)**2))/len(states)
"""
SNR = [100,10,5,2,1]
for i in SNR:
    data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
    clean = data["CleanStates"]
    X, y = buildMatrices(clean[:, 0], clean[:, 1], data["timestep"])
    pred = np.array([-1, -0.2])
    print(f"SNR {i}: loss at true pred = {loss(X, pred, y):.2e}")
    """

def grad(states,predicted,estimAccel,L2):
    return (states.T@(states@predicted-estimAccel))/(len(states)/2)+L2*2*predicted

def gradient_descent(params,epochs,lr,graphs,L2):
    true = np.array([params["alpha"],params["gamma"]])
    #-------------graphing
    lossCurve =[]
    alphaCurve = []
    gammaCurve = []
    #-------------
    pred = np.random.randn(2,)*np.sqrt(2)
    startLr = lr
    SNR = [100,10,5,2,1]
    for i in SNR:
        data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
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
            pred = pred-lr*grad(states_normal,pred,estimAccel,L2)
            alphaCurve.append(pred[0]/std[0])
            gammaCurve.append(pred[1]/std[1])
        print(f"The loss for SNR of {i} is {np.abs((true-(pred/std)*-1)/true)*100}% with the values of {(pred/std)*-1}")
        lr = startLr
        pred = np.random.randn(2,)*np.sqrt(2)
        #graphing-------------
        if graphs == True:
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

def table():
    plt.table()


if __name__ == "__main__":
    gradient_descent(linear_params,100,0.1,0,0.01) #params,epochs,lr,graphs,lambda(L2)

