from simulator import simulateRK4,easy_params,medium_params,hard_params
import numpy as np
from numpy import random


def SNR(displacement,SD):
    return np.std(displacement)/SD

def FDV(displacemet,timestep):
    estimateVel = np.diff(displacemet)*displacemet

