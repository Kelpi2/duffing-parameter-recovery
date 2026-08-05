from simulator import simulateRK4,easy_params,medium_params,hard_params,linear_params
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def addNoise(displacement,SD): #Guassian noise
    noise = np.random.normal(size = (displacement.shape), loc = 0,scale = SD)
    return displacement + noise

def MLPdataset(rows):
    training_Params = {"gamma":0,"alpha":0,"beta":0,"F":0,"omega":0}
    trainSet = np.zeros((rows,604))
    TruthLabels = np.zeros((rows,5))

    for traj in range(rows):
        training_Params["alpha"] = np.random.uniform(0.5, 2)
        training_Params["beta"] = np.random.uniform(0.2, 2)
        training_Params["gamma"] = np.random.uniform(0.1, 0.5)
        training_Params["F"] = np.random.uniform(0.5, 2)
        training_Params["omega"] = np.random.uniform(0.1, 0.6)
        states,__ = simulateRK4(38,0.063,training_Params,[1,0]) #TotTime,timestep(0.063),params,state
        trainSet[traj] = states[:604, 0]
        TruthLabels[traj,0] = training_Params["alpha"]
        TruthLabels[traj,1] = training_Params["beta"]
        TruthLabels[traj,2] = training_Params["gamma"]
        TruthLabels[traj,3] = training_Params["F"]
        TruthLabels[traj,4] = training_Params["omega"]
    np.savez(os.path.join(DATA_DIR, f"MLPdataset"), trainSet = trainSet, TruthLabels = TruthLabels)
    return trainSet,TruthLabels


def generateDataset(TotTime,timestep,params,state): #Generates and saves data,clean + noisy
    SNR = [100,10,5,2,1]
    states,__ = simulateRK4(TotTime,timestep,params,state)
    for i in SNR:
        noisyDis = addNoise(states[:,0],np.std(states[:,0])/i)
        noisyVel = FDV(noisyDis,timestep)
        noisyDis = noisyDis[1:-1]
        np.savez(os.path.join(DATA_DIR, f"dataset_SNR{i}"), CleanStates = states,NoisyDis = noisyDis,NoisyVel = noisyVel,timestep = timestep)

def FDV(displacement,timestep): #finite difference velocity
    estimatedVel = (displacement[2:]-displacement[:-2])/(2*timestep)
    return estimatedVel

def compare(TotTime,timestep,params,state): #compares 4 levels of SNR - Ignore
    generateDataset(TotTime,timestep,params,state)
    dis = []
    vel = []
    clean = 0
    SNR = [100,10,5,2,1]
    for i in SNR:
        data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
        dis.append(data["NoisyDis"])
        vel.append(data["NoisyVel"])
        clean = data["CleanStates"]
    fig,axs = plt.subplots(2,2)
    axs[0,0].plot(clean[:,0],clean[:,1])
    axs[0,0].set_title("Clean")
    axs[0,1].plot(dis[0],vel[0])
    axs[0,1].set_title("SNR:100")
    axs[1,0].plot(dis[1],vel[1])
    axs[1,0].set_title("SNR:10")
    axs[1,1].plot(dis[4],vel[4])
    axs[1,1].set_title("SNR:1")
    plt.show()

if __name__ == "__main__":
    #compare(1000,0.063,linear_params,[1,0])
    generateDataset(63,0.063,linear_params,[1,0]) #63 for 1000 data points