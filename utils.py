#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:14:34 2019

@author: rezwan
"""
###=========== First of all, we will import the needed dependencies 
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from matplotlib import pyplot as plt
from matplotlib import cm as cm
# plot the heatmap
import seaborn as sns

#### Seed
import random
seed = 42
random.seed(seed)


def correlation_matrix(df):

    plt.title('Feature Correlation')
    # calculate the correlation matrix
    corr = df.corr()
    # plot the heatmap
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)

def hist_feats(df):
    df.hist(figsize = (8,6))
    plt.show()

def DNN_model(X):
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(18, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
    
    # The Hidden Layers :
    NN_model.add(Dense(25, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(20, kernel_initializer='normal',activation='relu'))
    NN_model.add(Dense(15, kernel_initializer='normal',activation='relu'))
    
    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    
    # Compile the network :
    NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    
    return NN_model


import math

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_corr(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

##=========================Bland-Altman plot 
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random

def bland_altman_plot(data1, data2, name="", *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

#    plt.title('Bland-Altman Plot')
#    plt.legend()
    plt.scatter(mean, diff, *args, **kwargs, color='navy')
    md_plot = plt.axhline(md,           color='r', linestyle='--', label="md")
    md_plus = plt.axhline(md + 1.96*sd, color='g', linestyle='--', label="md + 1.96*sd")
    md_minus = plt.axhline(md - 1.96*sd, color='k', linestyle='--', label="md - 1.96*sd")
    plt.legend([md_plot, md_plus, md_minus], ['md', 'md + 1.96*sd', 'md - 1.96*sd'])
    
#    plt.xlabel("Average Hemoglobin(gm/dL)")
#    plt.ylabel("Difference Hemoglobin(gm/dL)")
    
    plt.savefig("imgs/"+ str(name)+".png", dpi = 100)
    plt.show()

def act_pred_plot(y, predicted, r=None, name=""):
    fig, ax = plt.subplots()
    ax.text(y.min(), y.max(), "Pearson's R = " + str('%.3f' %r))
    ax.scatter(y, predicted, color='navy')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=1)
#    ax.set_xlabel('Reference Hemoglobin(gm/dL)')
#    ax.set_ylabel('Estimated Hemoglobin(gm/dL)')
    plt.savefig("imgs/"+ str(name)+".png", dpi = 100)
    plt.show()

def rmse(targets,predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())
