# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:40:17 2019

@author: adityabiswas
"""

import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import RobustScaler

import GeneticFS


def fitnessFunc(input_tuple):
    i, X, Y = input_tuple
    trainPercent = 0.7
    n = len(Y)
    
    ## create marker for parameter-training / validation split
    marker = np.zeros(n, dtype = np.bool)
    cutoff = int(np.round(trainPercent*n))
    marker[:cutoff] = True
    np.random.shuffle(marker)
    
    ################################################################
    ### actual training and eval
    Xtrain, Xval = X[marker,:], X[~marker,:]
    Ytrain, Yval = Y[marker], Y[~marker]
    model = LR(C = 100, class_weight = 'balanced', solver = 'liblinear')
    model.fit(Xtrain, Ytrain)
    Pval = model.predict_proba(Xval)[:,1]
    auc = AUC(Yval, Pval)       
    return (i, auc)



if __name__ == "__main__":
    
    ## load
    baseFolder = 'G:\Projects\SomeProjectFolder'
    fileLoc = 'dataset.csv'
    data = pd.read_csv(os.path.join(baseFolder, fileLoc))

    ## separate into training marker, inputs, output
    marker = data.training_marker.values == 1
    Y = data.outcome.values == 1
    drop = ['training_marker', 'outcome', 'patient_id']
    X = data.drop(drop, axis = 1)
    features = X.columns.values
    
    ## split into training/test and normalize features
    Xtrain, Xtest = X.loc[marker,:], X.loc[~marker,:]
    Ytrain, Ytest = Y[marker], Y[~marker]
    scaler = RobustScaler()
    Xtrain = scaler.fit_transform(Xtrain.values)
    Xtest = scaler.transform(Xtest.values)
    
    
    ## set parameters
    numFeats = 10
    num_iters = 100
    popSize = 1000
    mutationRate = 0.05
    numClones = 2
    n_cores = 20
    
    freshStart = True
    namePopulation = 'population_LR_01012019'
    population = None if freshStart else namePopulation
    
    ## construct model
    print('Starting training')
    model = GeneticFS(fitnessFunc, Xtrain, otherData = (Ytrain), 
                        varNames = features, numFeats = numFeats,
                        population = population, folder = baseFolder, 
                        n_cores = n_cores, popSize = popSize, 
                        numClones = numClones, mutationRate = mutationRate)
    
    ## produce generations
    for i in range(num_iters):
        model.evolve() # next gen
        model.save(namePopulation) # checkpointing
    model.close()
    print(model.getBest())
    
    