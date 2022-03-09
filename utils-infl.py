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

eps = np.finfo(np.float64).eps
def sigmoid(z):
    return np.divide(1.0,1.0+np.clip(np.exp(-z),eps,np.infty))

def loss(theta,X,Y,n):
    z = np.dot(X,theta)
    h = sigmoid(z)
    return (-Y * np.log(h) - (1.0 - Y) * np.log(1.0 - h)).mean()

def logit_grad(theta,X,Y,n):
    z = np.dot(X,theta)
    h = sigmoid(z)
    return np.dot(X.T, (h - Y)) / n

def logit_hessian(theta,X,sigm,n):
    #z = np.dot(X,theta)
    #h = sigmoid(z)
    D = sigm*(1-sigm)/np.float(n)
    hess_R = np.multiply(np.transpose(X),D).dot(X)
    condition_n = np.linalg.cond(hess_R)
    if condition_n>30:
#         warnings.warn("Condition number is > 30")
#         print("Condition number: ",condition_n)
        return(None)
    return(hess_R)#+eps*np.identity(X.shape[1])

def influence_logit(x_tst, X_train, Y_train, X_train_h, Y_train_h, hess_R, theta, sigm, h_idx, n_train, print_det = False):
    grad_h = logit_grad(theta, X_train_h, Y_train_h, n_train)
    #hess_R = logit_hessian(theta,X_train, sigm, n_train)
    if hess_R is None:
        return None
    h = sigmoid(np.dot(x_tst,theta))
    grad_p = h*(1-h)*x_tst
    #print(grad_p)
#    if print_det:
#         print('determinant: ',np.linalg.det(hess_R))
#         print('condition number: ',np.linalg.cond(hess_R))
    #print(grad_p.shape,hess_R.shape,grad_h.shape)
    Hess_x_gradh = np.linalg.solve(hess_R,grad_h)
    return -np.transpose(grad_p).dot(Hess_x_gradh)

#Functions to measure and plot influence 
def influence_h(X_train, Y_train, X_test, X_train_h, Y_train_h, hess_R, theta, idx, sigm, n_train, n_tst):
    infl = np.zeros(n_tst)
    j=0
    for t in np.arange(n_tst):
        x_tst = X_test[t,:]
        infl[j] = influence_logit(x_tst, X_train, Y_train,  X_train_h, Y_train_h, hess_R, theta, sigm, idx, n_train)
        j+=1
#         if j % 5000 == 0:
#             print(j)
    return infl

#This function returns a matrix where each row corresponds to a decision-maker and each column to a test point
def influence_matrix(X_train, Y_train, X_test, hess_R, sigm, theta, screeners, screener_ids_train, print_update=False):
    n_train = X_train.shape[0]
    n_tst = X_test.shape[0]
    M_inf = np.zeros([len(screeners), n_tst])
#     z = np.dot(X_train,theta)
#     sigm = sigmoid(z)
    for i in np.arange(0,len(screeners)):
        #print(screeners[i])
        h_idx = np.array(screener_ids_train)==screeners[i]
        X_train_h = X_train[h_idx,:]
        Y_train_h = Y_train[h_idx]
        M_inf[i,:] = influence_h(X_train, Y_train, X_test, X_train_h, Y_train_h, hess_R, theta,  h_idx, sigm, n_train,n_tst)
        if print_update==True:
            print(i,len(screeners))
    return M_inf


# # Agreeability metrics

# In[20]:


#are less than k decision-makers responsible for more than pi of the influence?
def center_mass(inf_vec):
    inf_sorted = np.sort(np.abs(inf_vec))[::-1]
    #print(inf_sorted)
    center = np.dot(inf_sorted,np.arange(len(inf_vec)))/sum(inf_sorted)
    if center==0:
        print(inf_sorted)
    #print(center)
    return(center)
    #return(center/np.float(len(inf_vec)))
def opposing(inf_vec):
    inf_pos = inf_vec[np.where(inf_vec>0)]
    #print("pos ", inf_pos)
    inf_neg = inf_vec[np.where(inf_vec<0)]
    #print("neg ", inf_neg)
    inf_pos_sorted = np.sort(inf_pos)[::-1]
    #print("sort pos ", inf_pos_sorted)
    inf_neg_sorted = np.sort(inf_neg)
    #print("sort_neg ", inf_neg_sorted)
    k = min(len(inf_pos_sorted),len(inf_neg_sorted))
    if k ==0:
        m_agree=1.0
    else: 
        m_opposing = sum(inf_pos_sorted[:k])-sum(inf_neg_sorted[:k])
        m_agree = max(sum(inf_pos_sorted[:k]),np.abs(sum(inf_neg_sorted[:k])))/np.float(m_opposing)
    if m_agree==0:
        print("Found a zero! ", k, m_opposing)
    return(m_agree)


# In[23]:


def label_agreeability(X_train,Y_h_train,screener_ids_train, rho, C_grid = [0.05, 0.01, 0.005, 0.001]):
    skf = sklearn.model_selection.StratifiedKFold(n_splits=3)
    skf.get_n_splits(X_train, Y_h_train)
    screeners_set = [x for x in set(screener_ids_train) if str(x)!='nan']

    Y_pred_h = np.zeros((1,len(Y_h_train)))
    Agr1 = np.zeros((1,len(Y_h_train)))
    Agr2 = np.zeros((1,len(Y_h_train)))
    screeners_set = [x for x in set(screener_ids_train) if str(x)!='nan']
    M_inf = np.zeros((len(set(screeners_set)),len(Y_h_train)))
    #print(Y_pred_h)
    #print()
    
    for train_idx, test_idx in skf.split(X_train,Y_h_train):
        X_cvi_train, X_cvi_test = X_train[train_idx], X_train[test_idx]
        Y_h_cvi_train, Y_h_cvi_test = Y_h_train[train_idx], Y_h_train[test_idx]
        screener_ids_train_cvi = screener_ids_train[train_idx]
        hess_R = None
        
        i=0
        
        while hess_R is None:
            C=C_grid[i]
            
            logit = linear_model.LogisticRegression(penalty = 'l1', C=C, solver = 'liblinear')#,n_jobs=40)
            clf = logit.fit(X_cvi_train, Y_h_cvi_train)
            theta= np.array(list(clf.coef_[0]))
            
            theta_nonzero = theta[theta!=0]
            X_cvi_train_nonzero = X_cvi_train[:,theta!=0]
            n_train = X_cvi_train.shape[0]
            z = np.dot(X_cvi_train_nonzero,theta_nonzero)
            sigm = sigmoid(z)
            hess_R = logit_hessian(theta_nonzero,X_cvi_train_nonzero,sigm,n_train)
            i+=1
        print(C)
        print("non-zero coeff", sum(theta!=0))
        Y_pred_cvi = clf.predict_proba(X_cvi_test)[:, 1]
        
        
        clf_D_calibrated = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
        clf_D_calibrated.fit(X_cvi_test, Y_h_cvi_test)

        Y_pred_cvi = clf_D_calibrated.predict_proba(X_cvi_test)[:, 1]
        
#         clf_sigmoid = CalibratedClassifierCV(clf, cv=10, method='sigmoid')
#         clf_sigmoid.fit(X_cvi_train, Y_h_cvi_train)
#         Y_pred_cvi = clf_sigmoid.predict_proba(X_cvi_test)[:, 1]
        X_cvi_test = X_cvi_test[:,theta!=0]
        Y_pred_h[0,test_idx] = Y_pred_cvi
        idx_conf = np.where((Y_pred_cvi<rho) | (Y_pred_cvi>(1.0-rho)))
        idx_conf = np.array(idx_conf)[0]
        #print('conf cases: ',len(idx_conf))
        M_inf_cvi = influence_matrix(X_cvi_train_nonzero, Y_h_cvi_train, X_cvi_test[idx_conf,:], hess_R, sigm, theta_nonzero, screeners_set, screener_ids_train_cvi)
        if len(idx_conf)>0:
            Agr1_cvi_conf = np.apply_along_axis(center_mass,0,M_inf_cvi)
            Agr2_cvi_conf = np.apply_along_axis(opposing,0,M_inf_cvi)
            np.where(Agr2_cvi_conf==0)
            M_inf[:,test_idx[idx_conf]] = M_inf_cvi
            Agr1[0,test_idx[idx_conf]] = Agr1_cvi_conf
            Agr2[0,test_idx[idx_conf]] = Agr2_cvi_conf
    return M_inf, Y_pred_h, Agr1, Agr2



def label_h(X, screener_ids, screeners_set, coef_pred_y, change_coef=False, change_same = False, change_all = False, n=0, shared_bias = False, rand = False, bias_opposite = False, bias_assignment=False, random_if_not_good = False):
    D = np.zeros(X.shape[0])
    alphas = np.zeros(len(screeners_set))
    if change_same: #choose which coef to resample for all h
        print('change_same',n)
        
        #idx_change = random.sample(range(len(coef_pred_y)),n)
        #print(np.where(coef_pred_y!=0)[0])
        idx_change = random.sample(list(np.where(coef_pred_y!=0)[0]),n)
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
            coef[idx_change] = np.random.uniform(-max(np.abs(coef_pred_y)),max(np.abs(coef_pred_y)), size=n)
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
        #pr = np.dot(X[h_idx,:],coef)
       # print(np.mean(pr),np.std(pr))
        z=np.dot(X[h_idx,:],coef)+np.random.logistic(loc=0.0, scale=0.1,size=sum(h_idx))
        th = np.random.normal(0,0.1)
        
        D[h_idx] = z>th
        print(sum(D[h_idx])/len(D[h_idx]))
        
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