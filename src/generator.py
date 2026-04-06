from simulator import simulateRK4,easy_params,medium_params,hard_params
import numpy as np
import matplotlib.pyplot as plt
from numpy import random


def addNoise(displacement,SD):
    noise = np.random.normal(size = (len(displacement)), loc = 0,scale = SD)
    return displacement+noise

def generateDataset(TotTime,timestep,params,state):
    SNR = [100,10,5,2,1]
    for i in SNR:
        states,__ = simulateRK4(TotTime,timestep,params,state)
        noisyDis = addNoise(states[:,0],np.std(states[:,0])/i)
        noisyVel = FDV(noisyDis,timestep)
        noisyDis = noisyDis[1:-1]
        np.savez(f"C:\VS-Code Main\duffing-parameter-recovery\data\dataset_SNR{i}", CleanStates = states,NoisyDis = noisyDis,NoisyVel = noisyVel)

def FDV(displacement,timestep):
    estimatedVel = (displacement[2:]-displacement[:-2])/(2*timestep)
    return estimatedVel

def compare(TotTime,timestep,params,state):
    generateDataset(TotTime,timestep,params,state)
    dis = []
    vel = []
    clean = 0
    SNR = [100,10,5,2,1]
    for i in SNR:
        data = np.load(f"C:\VS-Code Main\duffing-parameter-recovery\data\dataset_SNR{i}.npz")
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

compare(1000,0.063,easy_params,[1,0])

