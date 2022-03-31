import numpy as np
import csv
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import random
from scipy.optimize import check_grad
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import numdifftools as nd
import collections
import pickle as pkl
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
import warnings
import itertools
from scipy.stats.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import normalized_mutual_info_score
import time



def label_h(X, screener_ids, screeners_set, coef_pred_y, change_coef=False, change_same = False, change_all = False, n=0, shared_bias = False, rand = False, bias_opposite = False, bias_assignment=False, random_if_not_good = False):
    D = np.zeros(X.shape[0])
    alphas = np.zeros(len(screeners_set))
    if change_same: #choose which coef to resample for all h
        print('change_same',n)
        
        idx_change = random.sample(range(len(coef_pred_y)),n)
        #print(np.where(coef_pred_y!=0)[0])
        #idx_change = random.sample(list(np.where(coef_pred_y!=0)[0]),n)
    if shared_bias:
        print('shared bias')
        if coef_pred_y[-2]==0:
            coef_bias = max(np.abs(coef_pred_y))
        else:
            coef_bias = np.sign(coef_pred_y[-2])*max(np.abs(coef_pred_y))
        if bias_opposite:
            print('bias opp')
            if coef_pred_y[-2]==0:
                print("No sign for misused feature, coefficient is 0.")
            else:
                coef_bias = -np.sign(coef_pred_y[-2])*max(np.abs(coef_pred_y))
    for j in np.arange(0,len(screeners_set)):
        h_idx = np.array(screener_ids)==screeners_set[j]
        
        #sample new coefficient to introduce bias?
        coef = np.copy(coef_pred_y)
        #print(coef)
        #time.sleep(10)
        if change_same: # all h misues the same n features (in different ways)
            #print('change_same',n)
            #coef[idx_change] = np.random.uniform(-max(np.abs(coef_pred_y)),max(np.abs(coef_pred_y)), size=n)
            coef[idx_change] = np.random.uniform(-1,1, size=n)
        elif change_all: #all features are misused (differently) by all humans
            #print('change all')
            coef[coef!=0] = np.random.uniform(-1,1, size=sum(coef!=0))
        elif change_coef: #each human misuses n features (different for each h)
            #print('change coef')
            coef[random.sample(range(len(coef)),n)] = np.random.normal(size=n)
    
        if shared_bias:
            
            coef[-2] = coef_bias
        if bias_assignment and screeners_set[j]=='TNew': #the new decision maker is biased against one subpopulation, identified by coef[-2]
            print('bias assign')
            #print('here!')
            if bias_opposite:
                coef_bias_h = -2*max(np.abs(coef_pred_y))
            else:
                coef_bias_h = 2*max(np.abs(coef_pred_y))
            coef[-2] = coef_bias_h
        #print('after',coef)
        #time.sleep(10)
        pr = np.dot(X[h_idx,:],coef)
        #print(np.mean(pr),np.std(pr))
        z=np.dot(X[h_idx,:],coef)+np.random.logistic(loc=0.0, scale=0.5,size=sum(h_idx))
        th = np.random.normal(0,0.5)
        print(th)
        th=0
        D[h_idx] = z>th
        #print(sum(D[h_idx])/len(D[h_idx]))
        
#         z=np.dot(X[h_idx,:],coef)+np.random.normal(loc=0.0,scale=0.1,size=sum(h_idx))
#         pr = 1/(1+np.exp(-z))
#         th = np.random.beta(5,5)
        
        alphas[j] = th
        #D[h_idx] = pr>th
        
    print(sum(D)/len(D))    
    return D, alphas


    
                   
def decision_model(X, screener_ids, screeners_set, coef_pred_y, change_coef=False, change_same = False, change_all = False, n=0,  shared_bias = False, rand = False, bias_opposite = False, bias_assignment=False, random_if_not_good = False):
    if rand==False:
        D, alphas = label_h(X, screener_ids, screeners_set,  coef_pred_y, change_coef, change_same, change_all, n, shared_bias, rand, bias_opposite,  bias_assignment, random_if_not_good)
                            
        return D, alphas
    else:
        D = np.random.binomial(X.shape[0],0.5)
        return D, np.repeat(0,len(screeners_set))