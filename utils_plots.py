#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:41:16 2019

@author: rezwan
"""
## import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns; sns.set()
#### Seed
import random
seed = 42
random.seed(seed)


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

#    corr = df.corr()
#    fig, ax = plt.subplots(figsize=(size, size))
#    ax.matshow(corr)
#    plt.xticks(range(len(corr.columns)), corr.columns);
#    plt.yticks(range(len(corr.columns)), corr.columns);
    
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 10)
    
    ax=sns.heatmap(df.corr())
    
    
##=========================Bland-Altman plot 
import matplotlib.pyplot as plt
import numpy as np

#def bland_altman_plot(data1, data2, *args, **kwargs):
#    data1     = np.asarray(data1)
#    data2     = np.asarray(data2)
#    mean      = np.mean([data1, data2], axis=0)
#    diff      = data1 - data2                   # Difference between data1 and data2
#    md        = np.mean(diff)                   # Mean of the difference
#    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
#
#    plt.scatter(mean, diff, *args, **kwargs)
#    plt.axhline(md,           color='gray', linestyle='--')
#    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
#    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')



