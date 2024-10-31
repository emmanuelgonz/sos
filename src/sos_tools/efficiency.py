import numpy as np
#  Logistic function for computing eta values

def efficiency_resolution(T,k,datarray):
    resolution_eta = 1 / (1 + np.exp(k * (datarray - T)))       
    return resolution_eta