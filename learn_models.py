#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 00:28:57 2019

@author: rezwan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:18:30 2018

@author: rezwan
"""
"""
Source: 
    (1) https://github.com/rezwanh001/PPG/blob/master/ppg/learn.py


"""
#### Seed
import random
seed = 42
random.seed(seed)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers
from sklearn.metrics import r2_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from measrmnt_indices import *

'''
def LinReg():
    ## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    reg = LinearRegression()
    return reg

##https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
def LinSVR():
    svr_lin = SVR(kernel='linear', C=1e3)
    return svr_lin

def RbfSVR():
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    return svr_rbf

def PLS():
    #https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
    pls2 = PLSRegression(n_components=2)
    return pls2

def DTR():
    #https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
    regr_2 = DecisionTreeRegressor(max_depth=17)
    return regr_2

def MLPR():
    #https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    MLR = MLPRegressor(hidden_layer_sizes=(50, ), 
                 activation='relu', solver="adam")
    return MLR

def RFR():
    # Fitting the Random Forest Regression Model to the dataset
#    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 150, random_state = 42)
    return regressor
    
'''

def DNN(X):
    model = Sequential()

    # The Input Layer :
    model.add(Dense(100, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
    
    # The Hidden Layers :
    model.add(Dense(150, kernel_initializer='normal',activation='relu'))
    model.add(Dense(200, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(250, kernel_initializer='normal',activation='relu'))
    model.add(Dense(300, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    
    # The Output Layer :
    model.add(Dense(7, kernel_initializer='normal',activation='linear'))
    
    # Compile the network :
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MAE'])
    
    return model

    