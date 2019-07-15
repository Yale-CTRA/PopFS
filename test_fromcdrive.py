# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 19:50:19 2019

@author: adityabiswas
"""

import multiprocessing
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import skewtest, boxcox, expon
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import os
import sys
root = os.path.join(os.path.expanduser('~'), 'Projects')
sys.path.append(root)
sys.path.append('G:\\Projects\\Genetic')

from GeneticParallel import GeneticFeatureSelection

from Helper.preprocessing import imputeMeans, standardize, convertISO8601
from Helper.utilities import save, load, getPatientIndices

from Helper.utilities import showCoef

import xgboost as xgb
from operator import xor



#def modelFunc(data):
#    X, Y = data
#    model = LR(class_weight = 'balanced')
#    model.fit(X, Y)
#    return model
#
#def fitnessFunc(model, X, Y):
#    P = model.predict_proba(X)[:,1]
#    return AUC(Y, P)

#trainingSize = 3e5
#posKeep = 3e4
#negKeep = trainingSize - posKeep
#
#def randomSelect(indices, N, keep = None, percent = None):
#    """ Selects a number or percentage of the indices to keep
#    and then returns a boolean vector the size of N where those
#    indices are set to true"""
#    assert xor(keep is None, percent is None)
#    n = int(keep) if percent is None else int(np.round(percent*N))
#    np.random.shuffle(indices)
#    indices = indices[:n]
#    select = np.zeros(N, dtype = np.bool)
#    select[indices] = True
#    return select
#
#
## find indices for positive/negative examples
#select = Y if Y.dtype == np.bool else Y == 1
#posIndices = np.arange(len(Y))[select]
#negIndices = np.arange(len(Y))[~select]
#
#assert posKeep <= len(posIndices)
#assert negKeep <= len(negIndices)
#markerPos = randomSelect(posIndices, len(Y), keep = posKeep)
#markerNeg = randomSelect(negIndices, len(Y), keep = negKeep)
#marker = np.logical_or(markerPos, markerNeg)

def fitnessFunc(input):
    i, X, Y, pIndex = input
    trainPer = 0.7
    
    np.random.shuffle(pIndex)
    pIndexTest = pIndex[int(np.round(trainPer*len(pIndex))):,:2]
    testIndices = np.concatenate([np.array(range(row[0], row[1])) for row in pIndexTest])
    marker = np.ones(len(Y), dtype = np.bool)
    marker[testIndices] = False
    
    ###########################################################################
    ## actual training and eval
    dtrain = xgb.DMatrix(X[marker,:], label = Y[marker])
    dtest = xgb.DMatrix(X[~marker,:], label = Y[~marker])
    del X, Y
    
    param = {'max_depth': 4, 'eta': 0.01, 'silent': 1, 'objective': 'binary:logistic', 
         'min_child_weight': 500, 'colsample_bytree': 0.8, 'subsample': 0.8, 'scale_pos_weight': 1,
         'gamma': 0.1, 'nthread': 1}
    
    num_round = 1000
    bst = xgb.train(param, dtrain, num_round, xgb_model = None)
    Ptest = bst.predict(dtest)
    auc = AUC(dtest.get_label(), Ptest)
    return (i, auc)



if __name__ == "__main__":
    baseFolder = 'G:\Data\The Peds'
    fileloc = os.path.join(baseFolder, 'peds analysis 1-23-2019.feather')
    data = pd.read_feather(fileloc)
    
    #data.time = convertISO8601(data.time)
    ID = data.pat_mrn_id.values
    Y = data.outcome.values
    training = data.Selected == 1
    
    
    data.race = data.race == 'Other'
    data.ethnicity = data.ethnicity == 'Other'
    data.sex = data.sex == 'F'
    
    data.drop(['pat_mrn_id', 'pat_enc_csn_id', 'outcome', 'time',
               'Selected', 'aki_time'], axis = 1, inplace = True)
    
    data['bmi'] = data.weight*2.205*703/(data.height*data.height)
    data['pp'] = data.systolic - data.diastolic
    features = data.columns.values
    X = data[features].values.astype(np.float32)
    
    
    Xtrain = X[training,:]
    Ytrain = Y[training]
    pIndexTrain = getPatientIndices(ID[training])
#    scaler = StandardScaler()
#    X = scaler.fit_transform(X)
    
    namePopulation = 'population_xgboost_022719'
    print('Starting training')
    population = load(baseFolder, namePopulation)
    model = GeneticFeatureSelection(fitnessFunc, Xtrain, (Ytrain, pIndexTrain), features,
                                    n_cores = 35, popSize = 1000, population = population,
                                    numClones = 3)
    
    for i in range(5):
        model.evolve()
        model.save(baseFolder, namePopulation)
    model.close()
    print(model.getBest())
    
    
    
