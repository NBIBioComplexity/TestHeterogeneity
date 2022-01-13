import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
import copy
import scipy
import pickle

from correlated_dist import *

def get_test_prevention_corr(N=10000, A_k=0.1, A_avg=0.5, corr=0.5, sensitivity=1.0):
    T = 1000
    t = 0
    # Rescaling the activity according to the timestep size:
    A_avg /= T

    # Convert correlation to xi parameter:
    xi = (corr**2-np.sqrt(corr**2 - corr**4))/(2 * corr**2 - 1)

    test_freqs = []
    As = [] # Activities
    Rs = [] # "Infectivities" (susceptibility * infectiousness, really)

    already_tested = []

    for i in range(N):
        A = np.random.gamma(A_k, A_avg/A_k)
        R =  A**2 # Reproductive number scales as A^2
        As.append(A)
        Rs.append(R)
        already_tested.append(False) # No-one is tested positive initially
    test_freqs, corr_tf_a, k_out = generate_correlated_freqdist(As, xi, A_k, A_avg)

    total_possible_inf = np.sum(Rs) * T
    prevented_inf = 0

    while t < T:
        for i in range(N):
            if not already_tested[i]:
                if random.random() < test_freqs[i] and random.random() < sensitivity:
                    # Individual tested (positive)
                    prevented_inf += Rs[i] * (T-t)
                    already_tested[i] = True
        t += 1
    prevent_frac = prevented_inf/total_possible_inf
    print("With k =", A_k, ", sensitivity =", sensitivity, ", and xi =", xi,",", round(100*prevent_frac,2), "% was prevented and the correlation was", round(corr_tf_a,2))
    return prevent_frac, test_freqs, corr_tf_a

pf, tfs, corr = get_test_prevention_corr(sensitivity=0.75)  # Returns the mean reduction in R0 as well as the frequency distribution and degree of correlation between activity and frequency. 
