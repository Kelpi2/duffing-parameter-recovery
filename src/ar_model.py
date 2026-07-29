import numpy as np
import os 
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

#2nd order ar reg

def fit(disp):
    n=len(disp)
    y = disp[2:]
    X = np.array([disp[1:n-1],disp[0:n-2]])
    X = np.column_stack(X)
    scalar = np.linalg.inv(X.T@X)@(X.T@y) #a1,a2 respectively
    return scalar[0],scalar[1]

def predict():
    SNR = SNR = [100,10,5,2,1]
    for i in SNR:
            data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
            a1,a2  = fit(data["CleanStates"][:,0])
            h = data["timestep"]
            gamma = -np.log(-a2)/h
            freq = (1/h)*np.arccos(a1/(2*np.sqrt(-a2)))
            alpha = freq**2 + (gamma**2)/4
            print(f"a = {alpha}, y = {gamma}, freq = {freq}")

predict()

