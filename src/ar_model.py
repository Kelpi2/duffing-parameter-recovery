import numpy as np
import os 
import matplotlib.pyplot as plt
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

def recoverParam(dec):
    SNR = [100,10,5,2,1]
    for i in SNR:
            data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
            #(a1, a2), ___ = fit(data["CleanStates"][::dec,0], 2, 2)
            (a1, a2), ___ = fit(data["NoisyDis"][:dec], 2, 2)
            h = data["timestep"]*dec
            if a2>=0 or abs(a1/(2*np.sqrt(-a2)))>1: #check for invaldi values
                 print(np.nan)
                 continue
            gamma = -np.log(-a2)/h
            freq = (1/h)*np.arccos(a1/(2*np.sqrt(-a2)))
            alpha = freq**2 + (gamma**2)/4
            print(f"a = {alpha}, y = {gamma}, freq = {freq}")

def AIC(Maxp,dec):
    SNR = [100,10,5,2,1]
    for i in SNR:
        AIC = []
        data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
        for order in range(1,Maxp+1):
            AIC.append(fit(data["NoisyDis"][::dec],order,Maxp)[1])
            #AIC.append(fit(data["CleanStates"][:,0]order,Maxp)[1]) #clean states testing
        print(f"Smallest AIC at SNR {i} has the order {AIC.index(min(AIC))+1}")

def predict(steps,order,trainEnd,spacing):
    SNR = [100,10,5,2,1]
    for i in SNR:
        data = np.load(os.path.join(DATA_DIR, f"dataset_SNR{i}.npz"))
        disp = data["NoisyDis"]
        clean = data["CleanStates"][1:,0]
        dispFit = disp[:trainEnd]
        scalars,__ = fit(dispFit,order,order)
        allErrors = []
        for start in range(trainEnd,len(disp)-steps,spacing):
            predicting = clean[start:start+steps]
            predictions = list(disp[start-order:start])
            errors = []
            for step in range(steps):
                prediction = 0
                for p in range(order):
                    prediction += scalars[p]*predictions[-(p+1)]
                predictions.append(prediction)
                errors.append(predicting[step]-predictions[order+step])
            allErrors.append(errors)
        allErrors = np.array(allErrors)
        RMSE = np.sqrt(np.mean(allErrors**2,axis=0))
        plt.plot(range(1,steps+1),RMSE,label=f"SNR {i}")
    plt.xlabel("Steps ahead")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()
        #print(f"At SNR level {i} the prediction is {predictions[50:]}")
    #print(predicting[0]-predictions[50])             

                 
if __name__ == "__main__":
    predict(200,2,50,5)
