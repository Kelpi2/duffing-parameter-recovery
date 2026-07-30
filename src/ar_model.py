import numpy as np
import os 
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

#2nd order ar reg

def fit(disp,order,Maxp):
    n=len(disp)
    y = disp[Maxp:]
    X = np.column_stack([disp[Maxp-k : n-k] for k in range(1, order+1)])
    scalars = np.linalg.inv(X.T@X)@(X.T@y) 
    residuals = y-X@scalars
    rss = residuals@residuals
    AIC = len(y)*np.log(rss/len(y))+2*order
    return scalars,AIC

def recoverParam():
    SNR = [100,10,5,2,1]
    for i in SNR:
            data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
            #(a1, a2), ___ = fit(data["CleanStates"][::25,0], 2, 2)
            (a1, a2), ___ = fit(data["NoisyDis"][::25], 2, 2)
            h = data["timestep"]*25
            if a2>=0 or abs(a1/(2*np.sqrt(-a2)))>1: #check for invaldi values
                 print(np.nan)
                 continue
            gamma = -np.log(-a2)/h
            freq = (1/h)*np.arccos(a1/(2*np.sqrt(-a2)))
            alpha = freq**2 + (gamma**2)/4
            print(f"a = {alpha}, y = {gamma}, freq = {freq}")

def AIC(Maxp):
    SNR = [100,10,5,2,1]
    for i in SNR:
        AIC = []
        data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
        for order in range(1,Maxp+1):
            AIC.append(fit(data["NoisyDis"],order,Maxp)[1])
            #AIC.append(fit(data["CleanStates"][:,0]order,Maxp)[1]) #clean states testing
        print(f"Smallest AIC at SNR {i} has the order {AIC.index(min(AIC))+1}")

def predict(steps,order,s):
    SNR = [100,10,5,2,1]
    for i in SNR:
        data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
        #disp = data["NoisyDis"]
        disp = data["CleanStates"][:,0] #debug
        dispFit = disp[:s] 
        predicting = disp[s:s+steps]
        scalars,__ = fit(dispFit,order,order)
        predictions = list(dispFit[s-order:])
        RMSE = []
        for step in range(steps):
            prediction = 0 
            for p in range(order):
                prediction += scalars[p]* predictions[-(p+1)]
            predictions.append(prediction)
            RMSE.append(predicting[step]-predictions[order+step])
        
        #print(f"At SNR level {i} the prediction is {predictions[50:]}")
    #print(predicting[0]-predictions[50])             

                 
if __name__ == "__main__":
    predict(1,2)
