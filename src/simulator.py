import numpy as np
import matplotlib.pyplot as plt


params = {"gamma":0.3,"alpha":-1,"beta":1,"F":0.3,"omega":1}

def duffing(t, state, gamma,alpha,beta,F,omega):
    FO1 = state[1]
    FO2 = F*np.cos(omega*t) - (gamma*FO1 + alpha*state[0]+beta*state[0]**3)
    state = np.array([FO1,FO2])
    return state

def eulerStep(state,t,timestep,params):
    return state + timestep*duffing(t,state,**params)
    
def RK4(state,t, timestep,params):
    k1 = duffing(t,state,**params)
    k2 = duffing(t+timestep/2,state + k1*timestep/2,**params)
    k3 = duffing(t+timestep/2,state + k2*timestep/2,**params)
    k4 = duffing(t+timestep,state + k3*timestep,**params)
    return state + (timestep/6)*(k1+2*k2+2*k3+k4)


def simulateRK4(TotTime,timestep,params,state):
    time = np.arange(0,TotTime,timestep)
    states = np.zeros((len(time),2))
    states[0] = state
    for index in range(1,len(time)):
        states[index] = RK4(states[index-1],time[index-1],timestep,params)
    energy = 0.5*states[:,1]**2 + 0.5*states[:,0]**2
    return states,energy

def simulateEuler(TotTime,timestep,params,state):
    time = np.arange(0,TotTime,timestep)
    states = np.zeros((len(time),2))
    states[0] = state
    for index in range(1,len(time)):
        states[index] = eulerStep(states[index-1],time[index-1],timestep,params)
    energy = 0.5*states[:,1]**2 + 0.5*states[:,0]**2
    return states,energy


def compare(steps,timestep,startingState):
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

def anSolution(steps,timestep,state,alpha):
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

compare(1000,0.063,[1,0])
    




        

