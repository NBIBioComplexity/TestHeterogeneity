import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick

import datetime as dt
import numpy as np
import pandas as pd
import cycler
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy.interpolate import interp1d
import math
from tqdm import tqdm
import time

label_fz = 16
title_fz = 24
l_width = 2.5

pr_res = 5
true_rates = np.linspace(0.5, 1, pr_res+1)

cm = plt.cm.get_cmap('inferno')
color = plt.cm.inferno(np.linspace(0, 1, len(true_rates)+1))

fs_label = 16
parameters = {
    "figure.titlesize": fs_label+6,
    "axes.labelsize": fs_label,
    "axes.titlesize": fs_label+4,
    "xtick.labelsize": fs_label,
    "ytick.labelsize": fs_label, 
    "legend.fontsize": fs_label,
    "legend.title_fontsize": fs_label + 6,
    "lines.linewidth": 3, 
    "font.family":"serif", 
    "mathtext.fontset": "dejavuserif", 
    "figure.dpi": 300,

}
plt.rcParams.update(parameters)

mystyle = 'seaborn'
plt.style.use(mystyle)

fontsize=10
font = {'family' : 'sans',
        'weight' : 'normal',
        'size'   : fontsize}
plt.rc('font', **font)

def figuresave(name, path="figs_heterogeneity/", formats=["png", "pdf"]):
    [plt.savefig(f"{path}{name}.{f}", facecolor=None, transparent=False, bbox_inches='tight', pad_inches=0) for f in formats]

sens_lbl = "$s$" 
freq_lbl = "$f$"

### test charecteristics default values ###
tpr_def, tnr_def = 1, 1


### defining the functions used ###
def cond_prob(tpr_in, tnr_in, infect_state):
    """
        Generate conditional probabilities, given true infectious state.
        Note that TPR and TNR can be functionalized by using evaluated functions as input.
    """
    return infect_state*tpr_in + (not infect_state)*(1 - tnr_in)
    
qr = lambda N: (int(N), N%1)  ##  Returns integer and remainder part of N 
    
def prob_repeated_test(N, TI=1, tpr_set=tpr_def, tnr_set=tnr_def, infect_state=True, d=0):
    """
        Returns the probability of returning true from repeated (rapid) tests, given true state.
    """
    q, r = qr(N*(1-d))
    
    tpr, fpr = cond_prob(tpr_set, tnr_set, True), cond_prob(tpr_set, tnr_set, False)
    fnr, tnr = 1 - tpr, 1 - fpr
    
    rate = int(infect_state)*fnr + int(not infect_state)*tnr
    return 1 - ((1-r)*rate**(q) + r*rate**(q+1))

def G(x, N):
    """
        Asisting function G(x, N)
    """
    q, r = qr(N) # int(N), N%1
    return (1/2 + x/(1-x))*(1-x**q) + (r*(q+r/2)*(1-x) - q)*(x**q)

def ETD_and_D(N, TI=1, tpr_set=tpr_def, tnr_set=tnr_def, infect_state=True, d=0):
    """
        Return: Expected Time of Detection, AND Detection, given true state
    """
    q, r = qr(N*(1-d))
    fnr = 1 - cond_prob(tpr_set, tnr_set, True)
    
    # return (sum(list((n+1/2)*fnr**(n) for n in range(q))) + r*(q + r/2)*fnr**(q))*(1-fnr)*TR
    return G(fnr, N*(1-d))*(TI/(N)) 
    
def ETD_given_D(N, TI=1, tpr_set=tpr_def, tnr_set=tnr_def, infect_state=True, d=0):
    """
        Return: Expected Time of Detection, GIVEN Detection occours and true state
    """
    exp_time_detect = ETD_and_D(N, TI=TI, tpr_set=tpr_set, tnr_set=tnr_set, infect_state=infect_state, d=d)
    prob_detect = prob_repeated_test(N, TI=TI, tpr_set=tpr_set, tnr_set=tnr_set, infect_state=infect_state, d=d)
    return exp_time_detect/prob_detect + d*TI

def ETD(N, TI=1, tpr_set=tpr_def, tnr_set=tnr_def, infect_state=True, d=0):
    """
        Return expected detection time, given true state
    """
    exp_time_detect = ETD_and_D(N, TI=TI, tpr_set=tpr_set, tnr_set=tnr_set, infect_state=infect_state, d=d)
    prob_no_detect = (1-prob_repeated_test(N, TI=TI, tpr_set=tpr_set, tnr_set=tnr_set, infect_state=infect_state, d=d))
    return exp_time_detect + prob_no_detect*TI*(1-d) + d*TI

def rho(N, s): 
    """
        Return the expected reduction as function of screening frequency.
    """    
    q, r = qr(N)
    return (1 - (G(1-s, N)/N + (1-r*s)*(1-s)**q))

def reduction(N, TI=1, tpr_set=tpr_def, tnr_set=tnr_def, d=0):
    """
        Return the expected reduction, taking delay into account.
    """    
    tpr =  cond_prob(tpr_set, tnr_set, True)
    # return (1-d)*(1 - (G(1-tpr, N*(1-d))/(N*(1-d)) + (1-r*tpr)*(1-tpr)**q))
    return (1-d)*rho(N*(1-d), tpr)


### Plotting the reduction and delay-effect ###

## Setting ranges and true pos. rates ##
x = np.linspace(0.01, 1e2, int(1e4))
dx = (1e2-0.01)/int(1e4)
win_max=800
delay = 0.2
true_rates_tmp = [0.5, 1]
#color = plt.cm.inferno(np.linspace(0, 1, len(true_rates)+1))


## Set colours ##
dcol = 0.25
col0 = 0

# Normal order
col_multipliers = [1, 2, 2.8]
# Reversed order:
col_reversed = True
if col_reversed:
    col_multipliers = list(reversed(col_multipliers))

col1 = cm(col0 + col_multipliers[0]*dcol)
col2 = cm(col0 + col_multipliers[1]*dcol)
col3 = cm(col0 + col_multipliers[2]*dcol)

cols = [col3, col2]

### Expected detection time for spectrum of parameters (sensitivity) ###
fig, ax = plt.subplots(figsize=(5,3.5))
for i, j in enumerate(true_rates_tmp):
    z = np.vectorize(reduction)(x, tpr_set=j, tnr_set=tnr_def, d=delay)
    ax.plot(x[:win_max], z[:win_max], label=f"{sens_lbl}={j:.1f}, d=0.2", color=cols[i], linestyle="--", alpha=0.5)
    y = np.vectorize(reduction)(x, tpr_set=j, tnr_set=tnr_def, d=0)
    ax.plot(x[:win_max], y[:win_max], label=f"{sens_lbl}={j:.1f}, d=0.0", color=cols[i])

ax.axhline(1-delay, color=col1, alpha=0.3) #label=f"Saturation at delay {delay}$T_{{I}}$", 

zo_dots = 10
#ax.axhline(0.6, color="tab:blue", alpha=0.3) #label=f"Saturation at delay {delay}$T_{{I}}$", 
ax.scatter(2.5, reduction(2.5, tpr_set=1, tnr_set=tnr_def, d=0.2), color="tab:blue", zorder=zo_dots)
ax.scatter(1.25, reduction(1.25, tpr_set=1, tnr_set=tnr_def, d=0), color="tab:blue", zorder=zo_dots)

#ax.axhline(0.5, color="tab:orange", alpha=0.3) #label=f"Saturation at delay {delay}$T_{{I}}$", 
ax.scatter(2.45, reduction(2.45, tpr_set=0.5, tnr_set=tnr_def, d=0), color="tab:orange", zorder=zo_dots)
ax.scatter(1., reduction(1., tpr_set=1, tnr_set=tnr_def, d=0), color="tab:orange", zorder=zo_dots)

## Format plot style ##
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

ax.set_ylim(0, 1)
ax.set_xlim(left=0)

ax.set_xlabel(f"Test freq. {freq_lbl} [$T_{{I}}^{{-1}}$]")
ax.set_ylabel("Reduction $\\rho$")
ax.set_title("Expected reduction")
ax.legend(ncol=1, loc="lower right")

figuresave("reduction_delay")


### Plotting the derivative of reduction and delay effects ###
fig, ax = plt.subplots(figsize=(5,3.5))
for i, j in enumerate(true_rates_tmp):
    delay=0
    y = np.vectorize(reduction)(x, tpr_set=j, tnr_set=tnr_def, d=delay)
    ax.plot(x[:win_max-1], np.diff(y[:win_max])/dx, label=f"{sens_lbl}={j:.1f}, d={delay:.1f}", color=cols[i], alpha=0.4)#, label=f"\frac{{d}}{{d {frq_lbl}}}\\rho({freq_lbl})")
    
    delay=0.2
    z = np.vectorize(reduction)(x, tpr_set=j, tnr_set=tnr_def, d=delay)
    ax.plot(x[:win_max-1], np.diff(z[:win_max])/dx, label=f"{sens_lbl}={j:.1f}, d={delay:.1f}", color=cols[i], linestyle=":")#, label=f"\frac{{d}}{{d {frq_lbl}}}\\rho({freq_lbl})")
    
ax.set_ylabel("Derivative of reduction $\\frac{{d \\rho}}{{df}}$")
ax.legend()
ax.set_xlabel(f"Test freq. {freq_lbl} [$T_{{I}}^{{-1}}$]")
figuresave("reduction_delay_deriv")

