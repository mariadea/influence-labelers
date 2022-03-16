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

from utils import *

#parameters data amalgamation
pi_1 = 4.0
pi_2= 0.8
tau = 1.0
rho= 0.05
K=20

setting = 'post_selabels_opb_unobs5_K20_v5'

data_file = '../../../data/semi_synthetic/Data_semisynthetic_v5.pkl'

#DATA GENERATING PROCESS PARAMETERS

selective_labels = True
#noise = True
opb = True
opb_blind = False

unobservables = True
unobs_k = 5 #number of unobsevables, k features with largest coefficient

change_some_coef = False #resample some coefficients for each human?
change_same = False
n=44#how many coefficients to change if change_some_coef == True
shared_bias = False
bias_opposite = False #if shared_bias true, should the bias overestimate the importance of use it in the opposite direction?

bias_assignment = False
change_all_coef = False#resample all non-zero coefficients?

random_if_not_good = False

#If opb_out, modeled as a business rule?
business_rule = False

#selective labels? Do we only observe label when D=1?


#HUMAN DECISIONS MODEL PARAMETERS

rand = False #are decisions made by humans random?


#if rand ==True, following parameters are relevant:





def partition_data(X,  refer_ids, screener_ids, Y_human, Y_observed):
    refer_ids_train, refer_ids_test = sklearn.model_selection.train_test_split(np.unique(refer_ids),test_size = 0.25)

    X_train = X[[(r in refer_ids_train) for r in refer_ids],:]
    X_test = X[[(r in refer_ids_test) for r in refer_ids],:]

    Y_h_train = np.array(Y_human[[(r in refer_ids_train) for r in refer_ids]])
    Y_h_test = np.array(Y_human[[(r in refer_ids_test) for r in refer_ids]])

    Y_obs_train = np.array(Y_observed[[(r in refer_ids_train) for r in refer_ids]])
    Y_obs_test = np.array(Y_observed[[(r in refer_ids_test) for r in refer_ids]])

    screener_ids_train = np.array(np.array(screener_ids)[[(r in refer_ids_train) for r in refer_ids]])
    screener_ids_test = np.array(np.array(screener_ids)[[(r in refer_ids_train) for r in refer_ids]])

    Y_h_train = np.transpose(Y_h_train)   #[0]
    Y_h_test = np.transpose(Y_h_test)#[0]
    Y_obs_train = np.transpose(Y_obs_train)#[0]
    Y_obs_test = np.transpose(Y_obs_test)#[0]
    
    return(refer_ids_train, refer_ids_test,X_train,X_test,Y_h_train,Y_h_test,Y_obs_train,Y_obs_test,screener_ids_train,screener_ids_test)


# In[25]:


def amalgamate(X,  refer_ids, screener_ids, Y_human, Y_observed):
    #partition data
    refer_ids_train, refer_ids_test,X_train,X_test,Y_h_train,Y_h_test,Y_obs_train,Y_obs_test, screener_ids_train,screener_ids_test = partition_data(X,  refer_ids, screener_ids, Y_human, Y_observed)
    #calculate consistency through cv . in train fols
    #print(Y_h_train)
    M_inf, Y_pred_h, Agr1, Agr2 = label_agreeability(X_train,Y_h_train,screener_ids_train,0.05)
    high_agr = np.where((Agr2[0]> pi_2) & (Agr1[0]> pi_1))
    #amalgamate labels
    Y_amalg_train = np.copy(Y_obs_train)
    high_agr_correct = [idx for idx in high_agr[0] if np.abs(Y_pred_h[0][idx]-Y_h_train[idx]) < 0.05 ]
    Y_amalg_train[high_agr_correct] = (1-tau)*Y_obs_train[high_agr_correct]+tau*Y_h_train[high_agr_correct]
    #train model on observed labels (on portion that is screened-in)
    logit = linear_model.LogisticRegression(penalty = 'l2')
    clf_obs = logit.fit(X_train[Y_h_train==1,:], Y_obs_train[Y_h_train==1])
    Y_pred_obs_test = clf_obs.predict_proba(X_test)[:, 1]
    #train model on amalgamated labels
    logit = linear_model.LogisticRegression(penalty = 'l2')
    idx_amalg_train = np.unique(np.concatenate((np.where(Y_h_train==1)[0], np.array(high_agr_correct))))
    #print(idx_amalg_train)
    idx_amalg_train = idx_amalg_train.astype(int)
    clf_amalg = logit.fit(X_train[idx_amalg_train,:], Y_amalg_train[idx_amalg_train])
    Y_pred_amalg_test= clf_amalg.predict_proba(X_test)[:, 1]
    #train model on human decisions
    logit = linear_model.LogisticRegression(penalty = 'l2')
    clf_human = logit.fit(X_train, Y_h_train)
    Y_pred_h_test = clf_human.predict_proba(X_test)[:, 1]
    return(Y_pred_obs_test, Y_pred_amalg_test, Y_pred_h_test, refer_ids_train, refer_ids_test)


# In[26]:


def amalgamate_kruns(X,  refer_ids, screener_ids, Y_human, Y_observed, K):
#     M_results = [amalgamate(X,  refer_ids, screener_ids, Y_human, Y_observed) for i in np.arange(K)]
    M_results = []
    for i in np.arange(K):
        M_results.append(amalgamate(X,  refer_ids, screener_ids, Y_human, Y_observed))
    return(M_results)

        
if __name__=="__main__":
    print("running")
    print(setting)
    
    with open(data_file, 'rb') as handle:
        X,Y_1,Y_2,Y,D_0,refer_ids,screener_ids,coef_pred_y = pkl.load(handle)


    if not opb:
        Y = Y_1
    elif opb_blind and not business_rule:
        Y = np.array([((Y_1[i]==1)&(D_0[i]==0)) for i in np.arange(len(Y_1))])
        Y_2 = 1-D_0
        logit = linear_model.LogisticRegression(penalty = 'l1', C=0.01, random_state=42, fit_intercept=False)
        clf = logit.fit(X, Y)
        Y_pred = clf.predict_proba(X)
        fpr, tpr,thres = sklearn.metrics.roc_curve(Y, Y_pred[:,1])
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        #print(roc_auc)
        coef_pred_y = clf.coef_
        #print(sum(coef_pred_y[0]!=0))
        #plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
    elif opb_blind and business_rule:
        Y_2 = 1-D_0
        


    if unobservables: #delete one of the variables that receive a lot of weight
        X_obs = np.delete(X,np.argsort(coef_pred_y[0])[-(unobs_k+2):-2],1)
    else:
        X_obs = X
        
    if bias_assignment:
        screener_ids = np.array(screener_ids)
        screener_ids[X[:,-2] == max(X[:,-2])] = 'TNew'
        screener_ids=list(screener_ids)

    screener_set = np.array([x for x in set(screener_ids) if str(x)!='nan'])
    
    D, alphas = decision_model(X, screener_ids, screener_set, coef_pred_y[0], change_coef = change_some_coef, change_same = change_same, change_all=change_all_coef, n=n,  shared_bias=shared_bias, rand= rand, bias_opposite=bias_opposite, bias_assignment= bias_assignment, random_if_not_good = random_if_not_good)

    if opb and opb_blind and business_rule:
        D[D_0] = 0
        Y[D_0] = 0
    
#     with open('../../data/semi_synthetic/Y_human_'+setting+'.pkl', 'wb') as file:
#         pkl.dump([X,Y_1,Y_2,Y,D_0,refer_ids,screener_ids,coef_pred_y,D],file)
    print(sum(D)/len(D))   
    print(sum(D==Y)/len(D)   )
    M_results = [amalgamate(X_obs,  refer_ids, screener_ids, np.array(D), np.array(Y_1)) for i in np.arange(K)]


# In[ ]:


    with open('../../../data/semi_synthetic/M_results_Kruns_'+setting+'.pkl', 'wb') as file:
        pkl.dump([X,Y_1,Y_2,Y,D_0,refer_ids,screener_ids,coef_pred_y,D, M_results],file)