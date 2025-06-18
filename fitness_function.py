from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
from sklearn.metrics import f1_score, r2_score
from measrmnt_indices import *
#### Seed
import random
seed = 42
random.seed(seed)


class FitenessFunction:
    
    def __init__(self,n_splits = 5,*args,**kwargs):
        """
            Parameters
            -----------
            n_splits :int, 
                Number of splits for cv
            
            verbose: 0 or 1
        """
        self.n_splits = n_splits
    

    def calculate_fitness(self,x,y, model):
        cv_set = np.repeat(-1.,x.shape[0])
        skf = KFold(n_splits = self.n_splits ,shuffle=True, random_state=42)
        model = model########## Import model
        for train_index,test_index in skf.split(x,y):
            x_train,x_test = x[train_index],x[test_index]
            y_train,y_test = y[train_index],y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()
            model.fit(x_train,y_train)
            predicted_y = model.predict(x_test)
            cv_set[test_index] = predicted_y
        return pearsonr(y,cv_set)
        

    
    def calculate_fitness_2(self,X,y):
        
        row = X.shape[0]
        col = X.shape[1]
        feat = col
        
        DX = []
        DY = []
        
        import math
        import numpy as np
        for i in range(row):
          for j in range(col):
            Dy = (y[i] - y[j])
            sm = 0
            if Dy >= 0:
              sm = sm
              for k in range(feat):
                # print(i, j, k)
                sm += (X[i][k] - X[j][k])**2
        
            elif Dy < 0:
              sm = -sm
              for k in range(feat):
                # print(i, j, k)
                sm += (X[i][k] - X[j][k])**2
            
            ## Sum
            DY.append(Dy)
            DX.append(math.sqrt(sm / feat))
              
#        print(len(DY))
#        print(len(DX))
#        print("=============")
#        print(DX)
#        print(DY)
#        
        ###================ Calculate SDxDy
        DX_mean = np.mean(DX)
        #print(DX_mean)
        DY_mean = np.mean(DY)
        #print(DY_mean)
        
        sm_DX_DY = 0
        for i in range(len(DX)):
            sm_DX_DY += (DX[i] - DX_mean) * (DY[i] - DY_mean)
        
        SDxDy = sm_DX_DY / (feat-1)
#        print("SDxDy : " + str(SDxDy))
            
        ###===============Calculte SDx
        sm_DX = 0
        for i in range(len(DX)):
            sm_DX += (DX[i] - DX_mean)**2
            
        SDx = sm_DX / (feat-1)
#        print("SDx : " + str(SDx))
        
        ###========== Calculte SDy
        sm_Dy = 0
        for i in range(len(y)):
            sm_Dy += (DY[i] - DY_mean)**2
        
        SDy = sm_Dy / (feat - 1)
#        print("SDy : " + str(SDy))    
        
        #### Now Calculate corelation-Coefficient: R
        R = (SDxDy) / math.sqrt(SDx * SDy)
#        print("R : " +str(R)) 
        return R

