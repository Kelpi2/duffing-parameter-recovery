from mlp import loader,forward,accuracy
from linear_regression import expLinearReg
from ar_model import coefToParams,fit
import os
import numpy as np
from generator import addNoise
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")



def dataPrep(X,Y,SNR,Xstd,Xmean):
    X = X[16000:]
    truthLabels = Y[16000:,[0,2]]

    if SNR != "clean":
        SD = X[:,:604].std(axis=1, keepdims=True)/SNR
        X = addNoise(X,SD)

    X_AR = (X-Xmean)/Xstd
    X_604 = X_AR[:,:604]
    return X_AR,X_604,truthLabels

def ExpLoop(levels,FileName):
    data = np.load(os.path.join(DATA_DIR, f"{FileName}.npz"))
    (X,Y) = (data["trainSet"],data["TruthLabels"])
    for snr in levels:
        print(f"SNR level: {snr}")
        weights,biases,Xmean,Xstd,Ymean,Ystd = loader(f"{FileName}_SNR_{snr}")
        X_REG,X_604,truthLabels = dataPrep(X,Y,snr,Xstd,Xmean)
        __,activated = forward(X_604,weights,biases)
        print("MLP")
        accuracy(activated,truthLabels,Ymean,Ystd,0) #activated,Y,Ymean,Ystd,reg)
        LinearResults = np.zeros((4000,2))
        ARresults = np.zeros((4000,2))
        NanCount = 0
        NanMask = []
        for i in range(4000):
            alpha,gamma = expLinearReg(X_REG[i,::25],25)
            LinearResults[i,0] = alpha
            LinearResults[i,1] = gamma
            (a1, a2), ___ = fit(X_REG[i,::25], 2, 2)  
            alpha,gamma = coefToParams(a1,a2,0.063*25,0)   #a1,a2,h,printY
            if np.isnan(alpha):
                NanMask.append(False)
                NanCount += 1
            else:
                NanMask.append(True)
            ARresults[i,0] = alpha
            ARresults[i,1] = gamma 
        print("Linear Regression")
        accuracy(LinearResults,truthLabels,Ymean,Ystd,1)
        print("AR")
        accuracy(ARresults[NanMask],truthLabels[NanMask],Ymean,Ystd,1)
        print(f"Nan Count: {NanCount}")

def DecSweep(dec,SNR,FileName):  
    data = np.load(os.path.join(DATA_DIR, f"{FileName}.npz"))
    (X,Y) = (data["trainSet"],data["TruthLabels"])
    minDec = []
    for snr in SNR:
        print(f"__________________________________SNR {snr}__________________________________")
        __,__,Xmean,Xstd,Ymean,Ystd = loader(f"{FileName}_SNR_{snr}")
        X_REG,__,truthLabels = dataPrep(X,Y,snr,Xstd,Xmean)
        R2 = []
        r2Dec = []
        for i in dec:
            LinearResults = np.zeros((4000,2))
            for row in range(4000):
                alpha,gamma = expLinearReg(X_REG[row,::i],i)
                LinearResults[row,0] = alpha
                LinearResults[row,1] = gamma
            print(f"Dec: {i}")
            r2,__ = accuracy(LinearResults,truthLabels,Ymean,Ystd,1)
            R2.append(r2[0])
            r2Dec.append(i)
        minDec.append(r2Dec[np.argmax(R2)])
        
    print(minDec)

def ARDecSweep(dec,SNR,FileName):  
    data = np.load(os.path.join(DATA_DIR, f"{FileName}.npz"))
    (X,Y) = (data["trainSet"],data["TruthLabels"])
    minDec = []
    for snr in SNR:
        print(f"__________________________________SNR {snr}__________________________________")
        __,__,Xmean,Xstd,Ymean,Ystd = loader(f"{FileName}_SNR_{snr}")
        X_REG,__,truthLabels = dataPrep(X,Y,snr,Xstd,Xmean)
        R2 = []
        r2Dec = []
        for i in dec:
            NanCount = 0
            NanMask = []
            ARresults = np.zeros((4000,2))
            for row in range(4000):
                (a1, a2), ___ = fit(X_REG[row,::i], 2, 2)  
                alpha,gamma = coefToParams(a1,a2,0.063*i,0)   #a1,a2,h,printY
                if np.isnan(alpha):
                    NanMask.append(False)
                    NanCount += 1
                else:
                    NanMask.append(True)
                ARresults[row,0] = alpha
                ARresults[row,1] = gamma
            print(f"Dec: {i}")
            if NanCount<=400:
                r2,__ = accuracy(ARresults[NanMask],truthLabels[NanMask],Ymean,Ystd,1)
                R2.append(r2[0])
                r2Dec.append(i)

        minDec.append(r2Dec[np.nanargmax(R2)])
        
    print(minDec)


if __name__ == "__main__":
    #ExpLoop(["clean",100,10,5,2,1],"ComparisonSet")
    ARDecSweep(range(1,40),["clean",100,10,5,2,1],"ComparisonSet")