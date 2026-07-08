from generator import FDV
import numpy as np
import matplotlib.pyplot as plt
from simulator import easy_params,medium_params,hard_params,linear_params
from generator import generateDataset
import os
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def buildMatrices(noisyDis,noisyVel,timestep): #Build matrix for normal eqs
    estimAccel = FDV(noisyVel,timestep)
    noisyDis = noisyDis[1:-1]
    noisyVel = noisyVel[1:-1]
    X = [noisyDis,noisyVel]
    return np.column_stack(X), estimAccel   #X,y

def normalEq(X,y):
    solutions = (np.linalg.inv((X.T@X)))@(X.T@y)*-1
    return solutions[0], solutions[1]

def linearReg(params,study):
    SNR = [100,10,5,2,1]
    studyA = []
    studyG = []
    for i in SNR:
        data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
        X,y = buildMatrices(data["NoisyDis"],data["NoisyVel"],data["timestep"])
        alpha,gamma = normalEq(X,y)
        if study == True:
            alpha = (np.abs(alpha-params["alpha"]))/params["alpha"]*100
            gamma = (np.abs(gamma-params["gamma"]))/params["gamma"]*100
            studyA.append(alpha)
            studyG.append(gamma)
        else:
            print(f"""At SNR {i} the percentage error for Alpha is {(np.abs(alpha-params["alpha"])/params["alpha"])*100:.2f}%
At SNR {i} the percentage error for Gamma is {(np.abs(gamma-params["gamma"])/params["gamma"])*100:.2f}%\n""")
    return studyA,studyG
        
#------------------------Noise study----------------------------#
def NoiseStudy(TotTime,timestep,params,state):
    alpha = []
    gamma = []
    SNR = [100,10,5,2,1]
    for i in range(20):
        generateDataset(TotTime,timestep,params,state)
        a,g = linearReg(params,1)
        alpha.append(a)
        gamma.append(g)
    alpha = np.array(alpha).T
    gamma = np.array(gamma).T
    Amean = np.mean(alpha,axis =1)
    Astd = np.std(alpha,axis=1)
    Gmean = np.mean(gamma,axis =1)
    Gstd = np.std(gamma,axis=1)
    plt.figure()
    plt.title("Alpha errorbar")
    plt.xscale('log')
    plt.xticks([100,10,5,2,1], ['100','10','5','2','1'])
    plt.errorbar(SNR,Amean,yerr=Astd,fmt='o')
    plt.axhline(y=20,color = "r")
    plt.show()
    plt.figure()
    plt.title("Gamma errorbar")
    plt.xscale('log')
    plt.xticks([100,10,5,2,1], ['100','10','5','2','1'])
    plt.errorbar(SNR,Gmean,yerr=Gstd,fmt='o')
    plt.axhline(y=20,color = "r")
    plt.show()

if __name__ == "__main__":
    linearReg(linear_params,0)
    #NoiseStudy(50,0.063,linear_params,[1,0])