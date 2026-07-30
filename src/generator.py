from simulator import simulateRK4,easy_params,medium_params,hard_params,linear_params
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def addNoise(displacement,SD): #Guassian noise
    noise = np.random.normal(size = (len(displacement)), loc = 0,scale = SD)
    return displacement + noise

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