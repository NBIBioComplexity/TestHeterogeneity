import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import time
import math
import copy
import scipy
import pickle
import glob

from correlated_dist import *

def reduction_with_known_f(f, alpha):
    if f < 0:
        reduction = 0
    else:
        if f <= 1:
            reduction = f*alpha/2
        else:
            reduction = (-2 + alpha + 2* f* alpha + (1 - alpha)**np.floor(f) * (2 + alpha * (-1 + f * (-2 + f * alpha)) + alpha * np.floor(f) * (2 - 2 * f * alpha + alpha * np.floor(f))))/(2 * f * alpha)
    return reduction

def reduction_w_heterogeneity_semicorr(N=10000, k=0.2, alpha=0.5, A_avg=0.5, xi=0):
    activities = np.random.gamma(k, A_avg/k, N)
    fs, corr, k_out = generate_correlated_freqdist(activities, xi, k, A_avg)
    reduction = 0
    tot_red_possible = 0
    for i in range(len(fs)):
        reduction += activities[i]**2 * reduction_with_known_f(fs[i], alpha)
        tot_red_possible += activities[i]**2
    tot_reduction = reduction/tot_red_possible    
    return tot_reduction, corr
    

# Sample run with corr=0.5
corr = 0.25
xi = (corr**2-np.sqrt(corr**2 - corr**4))/(2 * corr**2 - 1)
red_loc, corr_loc = reduction_w_heterogeneity_semicorr(xi=xi, N=50000) # Returns reduction in R and correlation

print(f"The obtained reduction was {round(100*red_loc,2)}%")
print(f"And the measured correlation was {corr_loc}")
