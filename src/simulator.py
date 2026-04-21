import numpy as np
import matplotlib.pyplot as plt


#parameters
linear_params = {"gamma":0.2,"alpha":1,"beta":0,"F":0,"omega":1}
easy_params = {"gamma":0.2,"alpha":1,"beta":0,"F":0.2,"omega":1}
medium_params = {"gamma":0.2,"alpha":0.8,"beta":0.4,"F":0.6,"omega":0.28}
hard_params = {"gamma":0.2,"alpha":-1,"beta":1.5,"F":0.5,"omega":0.4}

def duffing(t, state, gamma,alpha,beta,F,omega):      #2 1st order equations
    FO1 = state[1]
    FO2 = F*np.cos(omega*t) - (gamma*FO1 + alpha*state[0]+beta*state[0]**3)
    state = np.array([FO1,FO2])
    return state

def eulerStep(state,t,timestep,params):  #ForwardEuler-Ignore
    return state + timestep*duffing(t,state,**params)
    
def RK4(state,t, timestep,params):     #RK4 steps simulator-unpacks already
    k1 = duffing(t,state,**params)
    k2 = duffing(t+timestep/2,state + k1*timestep/2,**params)
    k3 = duffing(t+timestep/2,state + k2*timestep/2,**params)
    k4 = duffing(t+timestep,state + k3*timestep,**params)
    return state + (timestep/6)*(k1+2*k2+2*k3+k4)


def simulateRK4(TotTime,timestep,params,state):  #Runs rk4, saves states and energy
    time = np.arange(0,TotTime,timestep)
    states = np.zeros((len(time),2))
    states[0] = state
    for index in range(1,len(time)):
        states[index] = RK4(states[index-1],time[index-1],timestep,params)
    energy = (0.5*states[:,1]**2) + (0.5*params["alpha"]*states[:,0]**2)+(0.25*params["beta"]*states[:,0]**4)
    return states,energy

def simulateEuler(TotTime,timestep,params,state): #Runs ForwardEuler - Ignore
    time = np.arange(0,TotTime,timestep)
    states = np.zeros((len(time),2))
    states[0] = state
    for index in range(1,len(time)):
        states[index] = eulerStep(states[index-1],time[index-1],timestep,params)
    energy = 0.5*states[:,1]**2 + 0.5*states[:,0]**2
    return states,energy


def compare(steps,timestep,startingState,params):  #compares error between truth and two simulators - Ignore
    RK4state,RK4energy = simulateRK4(steps,timestep,params,startingState)
    EulerState,EulerEnergy = simulateEuler(steps,timestep,params,startingState)
    TrueState,TrueEnergy = anSolution(steps,timestep,startingState,params["alpha"])
    RK4EnerygyError = np.mean(np.abs(TrueEnergy-RK4energy))
    EulerEnergyError = np.mean(np.abs(TrueEnergy-EulerEnergy))
    RK4Error = (np.mean(np.abs(RK4state-TrueState)))
    EulerError = (np.mean(np.abs(EulerState-TrueState)))
    print(f"The energy error for RK4 was {RK4EnerygyError} and the plot error was {RK4Error}\nThe energy error for euler was {EulerEnergyError} and the plot error was {EulerError}")
    print(np.max(np.abs((RK4energy-TrueEnergy)/TrueEnergy)))

    plt.title("Phase Portrait")
    plt.plot(RK4state[:,0],RK4state[:,1],label = "RK4")
    #plt.plot(EulerState[:,0],EulerState[:,1],label = "Euler")
    plt.plot(TrueState[:,0],TrueState[:,1],label = "True")
    plt.axhline(0)
    plt.axvline(0)
    plt.legend()
    plt.figure()
    time = np.arange(0,steps,timestep)
    plt.title("Energy Drift")
    plt.plot(time,RK4energy,label = "RK4")
    plt.plot(time,EulerEnergy,label = "Euler")
    plt.plot(time,TrueEnergy,label = "True")
    plt.legend()
    plt.show()

def anSolution(steps,timestep,state,alpha): #True analytical solution for comparison
    time = np.arange(0,steps,timestep)
    x = state[0]
    v = state[1]
    omega = np.sqrt(alpha)
    amp = np.sqrt(x**2 + (v/omega)**2)
    phaseOffset = np.arctan2(-v,omega*x)
    x = amp* np.cos(omega*time+phaseOffset)
    v = -amp*omega*np.sin(omega*time+phaseOffset)
    energy = 0.5*(v**2) +0.5*alpha*(x**2)
    return np.column_stack((x,v)),energy


def PlotRK4(steps,timestep,params,state):  #Plots generated points
    RK4graph,__ = simulateRK4(steps,timestep,params,state)
    plt.title("Phase Portrait")
    plt.plot(RK4graph[:,0],RK4graph[:,1])
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()

def omegaSweep(steps,timestep,params,state): #Sweeps for best omega value based on amplitude
    omegas = np.arange(0.5,2,0.05)
    maxD = []
    for i in omegas:
        params["omega"] = i
        RK4graph,__ = simulateRK4(steps,timestep,params,state)
        maxD.append(np.max(np.abs(RK4graph[:,0])))
    plt.figure()
    plt.title("x VS omega")
    plt.plot(omegas,maxD)
    plt.axvline(x=np.sqrt(params["alpha"]),label = "natural freq")
    plt.show()
    print(np.max(maxD))



#PlotRK4(1000,0.063,hard_params,[1,0])

#omegaSweep(1000,0.063,easy_params,[1,0])

    




        

