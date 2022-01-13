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

def rnMean(data,meanWidth):
    return np.convolve(data, np.ones(meanWidth)/meanWidth, mode='valid')

def rnTime(t,meanWidth):
    return t[math.floor(meanWidth/2):-math.ceil(meanWidth/2)+1]


### Importing and selecting data ###

df_pcr = pd.read_csv("overvaagningsdata-covid19-10122021-gr7a/Test_pos_over_time.csv", sep=";", decimal=",").drop(index=[610, 611])
df_ag = pd.read_csv("overvaagningsdata-covid19-10122021-gr7a/Test_pos_over_time_antigen.csv", sep=";", decimal=",")#.drop(index=[610, 611])

for df in [df_pcr, df_ag]:
    df.drop(df.tail(2).index, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    
date_min = "2020-06-01"
date_max = "2021-12-06"
date_range_pcr = (df_pcr.Date > date_min) & (df_pcr.Date < date_max)
date_range_ag = (df_ag.Date > date_min) & (df_ag.Date < date_max)
all_dates = df_pcr.loc[date_range_pcr, "Date"].values

for c in ["NewPositive", "NotPrevPos", "PrevPos", "Tested", "Tested_kumulativ"]:
    df_pcr[c] = df_pcr[c].replace("\.", "", regex=True).astype(int)
    df_ag[c] = df_ag[c].replace("\.", "", regex=True).astype(int)

df_pcr = df_pcr.loc[df_pcr.Date.isin(all_dates)]
df_ag = df_ag.loc[df_ag.Date.isin(all_dates)]

df_pcr["wnr"] = df_pcr['Date'].dt.isocalendar().week
df_pcr["yr"] = df_pcr['Date'].dt.isocalendar().year

df_ag["wnr"] = df_ag['Date'].dt.isocalendar().week
df_ag["yr"] = df_ag['Date'].dt.isocalendar().year

pcr_groups = df_pcr.groupby(["yr", "wnr"]).sum()["Tested"]
ag_groups = df_ag.groupby(["yr", "wnr"]).sum()["Tested"]

### Plotting danish testing data ###

dcol = 0.25
col0 = 0

# Set colours:
# Normal order
col_multipliers = [1, 2, 2.8]
# Reversed order:
col_reversed = True
if col_reversed:
    col_multipliers = list(reversed(col_multipliers))
col1 = cm(col0 + col_multipliers[0]*dcol)
col2 = cm(col0 + col_multipliers[1]*dcol)
col3 = cm(col0 + col_multipliers[2]*dcol)

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(range(len(pcr_groups)), list(pcr_groups), color=col3, label="PCR")# , tick_label=list(str(pcr_groups.index)))
ax.bar(range(len(ag_groups)), list(ag_groups), bottom=list(pcr_groups), color=col1, label="AG")#, tick_label=list(str(pcr_groups.index)))
plt.xticks(list(range(len(pcr_groups)))[::4], list(f"W{w}, {y}" for y,w in pcr_groups.index)[::4], rotation=90)
ax.set_xlim(min(range(len(pcr_groups)))-0.5, max(range(len(pcr_groups)))+0.5)
ax.legend()
figuresave("tests_DK_bar_aspect")

