#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:30:48 2018

@author: rezwan
"""
import math
def main():
    TP = float(input("Enter TP: "))
    FP = float(input("Enter FP: "))
    FN = float(input("Enter FN: "))
    TN = float(input("Enter TN: "))
    
    print("---------------------------------------------------")
    
    acc = ((TP+TN)/(TP+TN+FP+FN))
    print("Accuracy: " + str(acc))
    
    TPR = (TP/(TP + FN))
    print("Sensitivity,Recall,Hit Rate or True Positive Rate: " + str(TPR))
    
    TNR = (TN/(TN+FP))
    print("Specifity or True Negative Rate: " + str(TNR))
    
    PPV = (TP/(TP+FP))
    print("Precision or Positive Predictive Value: " + str(PPV))
    
    NPV = (TN/(TN+FN))
    print("Negative Predictive Value: " + str(NPV))
    
    FNR = 1.0 - TPR
    print("Miss Rate or False Negative Rate: " + str(FNR))
    
    FPR = 1.0 - TNR
    print("Fall-Out or False Positive Rate: " + str(FPR))
    
    FDR = 1.0 - PPV
#    print("False Discovery Rate: " + str(FDR))
    
    FOR = 1.0 - NPV
#    print("False Omission Rate: " + str(FOR))
    
    
    F1_Score = ((2.0*TP)/((2.0*TP)+FP+FN))
    print("F1 Score: " + str(F1_Score))
    
    MCC = (((TP*TN)-(FP*FN))/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))
    print("Matthews Correlation Coefficient: " + str(MCC))
    
    BM = TPR+TNR-1.0
#    print("Informedness of Bookmaker informedness: " + str(BM))
    
    MK = PPV+NPV-1.0
#    print("Markedness: " + str(MK))
    
main()