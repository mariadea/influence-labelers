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
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# In[2]:


#parameters
pi_1 = 4.0
pi_2= 0.8
tau = 1.0
rho= 0.02
K=10
class_choice = 'logit'
file_save = '../data/ChildWelfare/M_results_Kruns_hybridamalg_logit.pkl'





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

    Y_h_train = np.transpose(Y_h_train)[0]
    Y_h_test = np.transpose(Y_h_test)[0]
    Y_obs_train = np.transpose(Y_obs_train)[0]
    Y_obs_test = np.transpose(Y_obs_test)[0]
    
    return(refer_ids_train, refer_ids_test,X_train,X_test,Y_h_train,Y_h_test,Y_obs_train,Y_obs_test,screener_ids_train,screener_ids_test)


# In[25]:

def label_agreeability_test(X_train, X_test, Y_h_train, screener_ids_train, rho):
    
    screeners_set = [x for x in set(screener_ids_train) if str(x)!='nan']

    #Y_pred_h = np.zeros((1,len(Y_h_train)))
    
    Agr1 = np.zeros((1,X_test.shape[0]))
    Agr2 = np.zeros((1,X_test.shape[0]))
    screeners_set = [x for x in set(screener_ids_train) if str(x)!='nan']
    M_inf = np.zeros((len(set(screeners_set)),len(Y_h_train)))

    hess_R = None
    C_grid = [0.05, 0.01, 0.005, 0.001]
    i=0
    while hess_R is None:
        C=C_grid[i]
        logit = linear_model.LogisticRegression(penalty = 'l1', C=C, solver = 'liblinear')#,n_jobs=40)
        clf = logit.fit(X_train, Y_h_train)
        theta= np.array(list(clf.coef_[0]))
        #print("non-zero coeff", sum(theta!=0))
        theta_nonzero = theta[theta!=0]
        X_train_nonzero = X_train[:,theta!=0]
        n_train = X_train.shape[0]
        z = np.dot(X_train_nonzero,theta_nonzero)
        sigm = sigmoid(z)
        hess_R = logit_hessian(theta_nonzero,X_train_nonzero,sigm,n_train)
        i+=1    
        print('C', C)
        
    Y_pred_h = clf.predict_proba(X_test)[:, 1]
            
    clf_D_calibrated = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
    clf_D_calibrated.fit(X_train, Y_h_train)

    Y_pred_h = clf_D_calibrated.predict_proba(X_test)[:, 1]

    X_test = X_test[:,theta!=0]

    idx_conf = np.where((Y_pred_h<rho) | (Y_pred_h>(1.0-rho)))
    idx_conf = np.array(idx_conf)[0]
        #print('conf cases: ',len(idx_conf))
    M_inf = influence_matrix(X_train_nonzero, Y_h_train, X_test[idx_conf,:], hess_R, sigm, theta_nonzero, screeners_set, screener_ids_train)
    
    if len(idx_conf)>0:
        Agr1_conf = np.apply_along_axis(center_mass,0,M_inf)
        Agr2_conf = np.apply_along_axis(opposing,0,M_inf)
        Agr1[0,idx_conf] = Agr1_conf
        Agr2[0,idx_conf] = Agr2_conf
    return M_inf, Y_pred_h, Agr1, Agr2


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



def amalgamate_and_hybrid(X,  refer_ids, screener_ids, Y_human, Y_observed, fit_classifier ='logit', f_y_hyb = 'retrain'):
    #partition data
    refer_ids_train, refer_ids_test,X_train,X_test,Y_h_train,Y_h_test,Y_obs_train,Y_obs_test, screener_ids_train,screener_ids_test = partition_data(X,  refer_ids, screener_ids, Y_human, Y_observed)
    
    if fit_classifier =='logit':
        clf_obs = linear_model.LogisticRegression(penalty = 'l2', solver = 'liblinear')
        clf_amalg = linear_model.LogisticRegression(penalty = 'l2', solver = 'liblinear')
        clf_human = linear_model.LogisticRegression(penalty = 'l2', solver = 'liblinear')
    if fit_classifier =='rf':
        param_grid = {'min_samples_leaf':[10, 25, 50]}
        clf_obs = RandomForestClassifier(random_state=0)
        clf_amalg = RandomForestClassifier(random_state=0)
        clf_human = RandomForestClassifier(random_state=0)
        
    #calculate consistency through cv . in train fols
    M_inf, Y_pred_h, Agr1, Agr2 = label_agreeability(X_train,Y_h_train,screener_ids_train,0.05)
    high_agr = np.where((Agr2[0]> pi_2) & (Agr1[0]> pi_1))
    
    #amalgamate labels
    Y_amalg_train = np.copy(Y_obs_train)
    high_agr_correct = [idx for idx in high_agr[0] if np.abs(Y_pred_h[0][idx]-Y_h_train[idx]) < 0.05 ]
    Y_amalg_train[high_agr_correct] = (1-tau)*Y_obs_train[high_agr_correct]+tau*Y_h_train[high_agr_correct]
    
    #train model on observed labels (on portion that is screened-in)
    if fit_classifier =='logit':
        clf_obs.fit(X_train[Y_h_train==1,:], Y_obs_train[Y_h_train==1])
    if fit_classifier =='rf':
        grid_search = GridSearchCV(estimator = clf_obs, param_grid = param_grid, 
                          cv = 3, n_jobs = 10, verbose = 1)
        grid_search.fit(X_train[Y_h_train==1,:], Y_obs_train[Y_h_train==1])
        clf_obs = grid_search.best_estimator_
    
    Y_pred_obs_train = clf_obs.predict_proba(X_train[Y_h_train==1,:])[:, 1]
    Y_pred_obs_test = clf_obs.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(Y_obs_train[Y_h_train==1], Y_pred_obs_train)
    print("AUC_train f_y", metrics.auc(fpr, tpr))
    
    #train model on amalgamated labels
    idx_amalg_train = np.unique(np.concatenate((np.where(Y_h_train==1)[0], np.array(high_agr_correct))))
    idx_amalg_train = idx_amalg_train.astype(int)
    if fit_classifier =='logit':
        clf_amalg.fit(X_train[idx_amalg_train,:], Y_amalg_train[idx_amalg_train])
    if fit_classifier =='rf':
        grid_search = GridSearchCV(estimator = clf_amalg, param_grid = param_grid, 
                          cv = 3, n_jobs = 10, verbose = 1)
        grid_search.fit(X_train[idx_amalg_train,:], Y_amalg_train[idx_amalg_train])
        clf_amalg = grid_search.best_estimator_
    
    Y_pred_amalg_train = clf_amalg.predict_proba(X_train[idx_amalg_train,:])[:, 1]
    Y_pred_amalg_test= clf_amalg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(Y_amalg_train[idx_amalg_train], Y_pred_amalg_train)
    print("AUC_train f_A", metrics.auc(fpr, tpr))
    
    #train model on human decisions
    if fit_classifier =='logit':
        clf_human.fit(X_train, Y_h_train)
    if fit_classifier =='rf':
        grid_search = GridSearchCV(estimator = clf_human, param_grid = param_grid, 
                          cv = 3, n_jobs = 10, verbose = 1)
        grid_search.fit(X_train, Y_h_train)
        clf_human = grid_search.best_estimator_
    
    Y_pred_h_train = clf_human.predict_proba(X_train)[:, 1]
    Y_pred_h_test = clf_human.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(Y_h_train, Y_pred_h_train)
    print("AUC_train f_h", metrics.auc(fpr, tpr))
    
    #Determine which test instances are in set A
    M_inf_test, Y_pred_h_test_L1, Agr1_test, Agr2_test = label_agreeability_test(X_train, X_test, Y_h_train, screener_ids_train, rho)
    #idx of high_agreement instances
    high_agr_test = np.where((Agr2_test[0]> pi_2) & (Agr1_test[0]> pi_1))[0]
    
    
    #hybrid predictions: if x in A->f_h, if x not in A->f_y
    Y_pred_hybrid_test = np.copy(Y_pred_obs_test)
    
    
    #Assign predictions of human model to high-consistency set for hybrid approach 
    #uses same model as the one used to obtain set A_test
    if f_y_hyb=='retrain':
        Y_pred_hybrid_test[high_agr_test] = Y_pred_h_test[high_agr_test]
    else:
        Y_pred_hybrid_test[high_agr_test] = Y_pred_h_test_L1[high_agr_test]
    #uses separate human model (L2) to ensure comparability with amalgamated model
    
    
    
    return(Y_pred_hybrid_test, Y_pred_obs_test, Y_pred_amalg_test, Y_pred_h_test, refer_ids_train, refer_ids_test)

# In[26]:


def amalgamate_kruns(X,  refer_ids, screener_ids, Y_human, Y_observed, K):
#     M_results = [amalgamate(X,  refer_ids, screener_ids, Y_human, Y_observed) for i in np.arange(K)]
    M_results = []
    for i in np.arange(K):
        print('iteration',i)
        M_results.append(amalgamate_and_hybrid(X,  refer_ids, screener_ids, Y_human, Y_observed, class_choice))
    return(M_results)

        


# In[27]:

if __name__=="__main__":
    print("running")
    
    with open('../../../data/ChildWelfare/X_preprocess.pkl', 'rb') as handle:
        X,screener_ids,refer_ids,Y_observed,Y_human,Y_serv,Y_sub,colnames = pkl.load(handle)
    M_results = amalgamate_kruns(X,  refer_ids, screener_ids, Y_human, Y_observed, K)
    #M_results = [amalgamate(X,  refer_ids, screener_ids, Y_human, Y_observed) for i in np.arange(K)]


# In[ ]:


    with open(file_save, 'wb') as file:
        pkl.dump(M_results,file)