from mlp import loader,forward,accuracy
from linear_regression import expLinearReg
from ar_model import coefToParams,fit
import matplotlib.pyplot as plt
import os
import numpy as np
from generator import addNoise
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")



def dataPrep(X,Y,SNR,Xstd,Xmean,Test):
    np.random.seed(200)
    if Test:
        X = X[16000:]
        truthLabels = Y[16000:,[0,2]]
    else:
        X = X[12000:16000]
        truthLabels = Y[12000:16000,[0,2]]
        
    if SNR != "clean":
        SD = X[:,:604].std(axis=1, keepdims=True)/SNR
        X = addNoise(X,SD)

    Xnorm = (X-Xmean)/Xstd
    X_604 = Xnorm[:,:604]
    X_AR = X[:,:604]
    return X_AR,X_604,truthLabels

def ExpLoop(levels,FileName,Test,save):
    AlphalinearDec = [1, 2, 5, 6, 8, 10]
    AlphaArDec = [1, 20, 22, 22, 23, 28]
    GammalinearDec = [1, 1, 3, 3, 2, 2]
    GammaArDec = [1, 17, 23, 24, 23, 26]
    data = np.load(os.path.join(DATA_DIR, f"{FileName}.npz"))
    (X,Y) = (data["trainSet"],data["TruthLabels"])
    ind = 0
    stds = np.std(Y[:,[0,2]],axis=0)
    for param in ["alpha","gamma"]:
        LRmse = []
        ARmse = []
        MLPRmse = []
        for snr in levels:
            print(f"SNR level: {snr}")
            weights,biases,Xmean,Xstd,Ymean,Ystd = loader(f"{FileName}_SNR_{snr}")
            X_REG,X_604,truthLabels = dataPrep(X,Y,snr,Xstd,Xmean,Test)
            __,activated = forward(X_604,weights,biases)
            print("MLP")
            __,rmse,absMed = accuracy(activated,truthLabels,Ymean,Ystd,0) #activated,Y,Ymean,Ystd,reg)
            MLPRmse.append(rmse[ind])
            print(f"Absolute Median Error: {absMed}")
            LinearResults = np.zeros((4000,2))
            ARresults = np.zeros((4000,2))
            NanCount = 0
            if param == "alpha":
                LDec = AlphalinearDec[levels.index(snr)]
                ADec = AlphaArDec[levels.index(snr)]
            else:
                LDec = GammalinearDec[levels.index(snr)]
                ADec = GammaArDec[levels.index(snr)]
            NanMask = []
            for i in range(4000):
                alpha,gamma = expLinearReg(X_REG[i,::LDec],LDec)
                LinearResults[i,0] = alpha
                LinearResults[i,1] = gamma
                (a1, a2), ___ = fit(X_REG[i,::ADec], 2, 2)  
                alpha,gamma = coefToParams(a1,a2,0.063*ADec,0)   #a1,a2,h,printY
                if np.isnan(alpha):
                    NanMask.append(False)
                    NanCount += 1
                else:
                    NanMask.append(True)
                ARresults[i,0] = alpha
                ARresults[i,1] = gamma 
            print("Linear Regression")
            __,rmse,absMed = accuracy(LinearResults,truthLabels,Ymean,Ystd,1)
            LRmse.append(rmse[ind])
            print(f"Absolute Median Error: {absMed}")
            print("AR")
            __,rmse,absMed= accuracy(ARresults[NanMask],truthLabels[NanMask],Ymean,Ystd,1)
            ARmse.append(rmse[ind])
            print(f"Absolute Median Error: {absMed}")
            print(f"Nan Error: {NanCount/4000:.3f}%")

        snrLevels = [str(l) for l in levels[1:]]
        plt.figure()
        plt.yscale("log")
        plt.ylabel("RMSE")
        plt.xlabel("SNR")
        plt.title(f"{param} Recovery: RMSE")
        plt.plot(snrLevels,MLPRmse[1:],label = "MLP")
        plt.axhline(stds[ind],linestyle = "--",label = f"{param} std")
        plt.plot(snrLevels,LRmse[1:],label = "Linear Reg")
        plt.plot(snrLevels,ARmse[1:],label ="AR Reg")
        plt.legend()
        if save:
                    plt.savefig(os.path.join(FIG_DIR,f"{param} Recovery Comparison.png"))
        plt.show()
        ind +=1
        


def DecSweep(dec,SNR,FileName,Test,ay):  
    data = np.load(os.path.join(DATA_DIR, f"{FileName}.npz"))
    (X,Y) = (data["trainSet"],data["TruthLabels"])
    minDec = []
    for snr in SNR:
        print(f"__________________________________SNR {snr}__________________________________")
        __,__,Xmean,Xstd,Ymean,Ystd = loader(f"{FileName}_SNR_{snr}")
        X_REG,__,truthLabels = dataPrep(X,Y,snr,Xstd,Xmean,Test)
        R2 = []
        r2Dec = []
        for i in dec:
            LinearResults = np.zeros((4000,2))
            for row in range(4000):
                alpha,gamma = expLinearReg(X_REG[row,::i],i)
                LinearResults[row,0] = alpha
                LinearResults[row,1] = gamma
            print(f"Dec: {i}")
            r2,__,__ = accuracy(LinearResults,truthLabels,Ymean,Ystd,1)
            R2.append(r2[ay])
            r2Dec.append(i)
        minDec.append(r2Dec[np.argmax(R2)])
        
    print(minDec)

def ARDecSweep(dec,SNR,FileName,Test,ay):  
    data = np.load(os.path.join(DATA_DIR, f"{FileName}.npz"))
    (X,Y) = (data["trainSet"],data["TruthLabels"])
    minDec = []
    for snr in SNR:
        print(f"__________________________________SNR {snr}__________________________________")
        __,__,Xmean,Xstd,Ymean,Ystd = loader(f"{FileName}_SNR_{snr}")
        X_REG,__,truthLabels = dataPrep(X,Y,snr,Xstd,Xmean,Test)
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
            print(NanCount)
            if NanCount<=400:
                r2,__,__ = accuracy(ARresults[NanMask],truthLabels[NanMask],Ymean,Ystd,1)
                R2.append(r2[ay])
                r2Dec.append(i)

        minDec.append(r2Dec[np.nanargmax(R2)])
        
    print(minDec)


if __name__ == "__main__":
    ExpLoop(["clean",100,10,5,2,1],"ComparisonSet",0,1)
    #ARDecSweep(range(1,40),["clean",100,10,5,2,1],"ComparisonSet",0,1) #dec,SNR,FileName,Test,ay(0=alpha,1=gamma)