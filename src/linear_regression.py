from generator import FDV
import numpy as np
import matplotlib.pyplot as plt
from simulator import easy_params,medium_params,hard_params,linear_params


def buildMatrices(noisyDis,noisyVel,timestep): #Build matrix for normal eqs
    estimAccel = FDV(noisyVel,timestep)
    noisyDis = noisyDis[1:-1]
    noisyVel = noisyVel[1:-1]
    X = [noisyDis,noisyVel]
    return np.column_stack(X), estimAccel   #X,y

def normalEq(X,y):
    solutions = (np.linalg.inv((X.T@X)))@(X.T@y)*-1
    return solutions[0], solutions[1]

def linearReg(params):
    SNR = [100,10,5,2,1]
    for i in SNR:
        #data = np.load(f"C:\VS-Code Main\duffing-parameter-recovery\data\dataset_SNR{i}.npz") #dekstop
        data = np.load(f"C:\VS-Code\duffing-parameter-recovery\data\dataset_SNR{i}.npz") #laptop
        X,y = buildMatrices(data["NoisyDis"],data["NoisyVel"],data["timestep"])
        alpha,gamma = normalEq(X,y)
        print(f"""For an SNR of {i} the calculated value of Alpha is {alpha:.2f} and the true value is {params["alpha"]} with a percentage accuracy of {(np.abs(alpha-params["alpha"])/params["alpha"])*100:.2f}%
For an SNR of {i} the calculated value of Gamma is {gamma:.2f} and the true value is {params["gamma"]} with a percentage accuracy of {(np.abs(gamma-params["gamma"])/params["gamma"])*100:.2f}%\n""")

#linearReg(linear_params)
