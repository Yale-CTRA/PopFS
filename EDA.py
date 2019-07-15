# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:26:48 2019

@author: adityabiswas
"""


import numpy as np
import scipy as sci
import multiprocessing
import pickle
from collections import defaultdict
import statistics
import math

class PBIL(object):
    def __init__(self, fitnessFunc, X, otherData, varNames,
                 popSize = 100, lr = 0.5, tol = 0.05, init_prob = 0.5,
                 n_cores = 1, probabilities = None, verbose = True):
        """
        X = numpy ndarray containing the all the input feature data
        otherData = a tuple containing any other relevant data to your fitnesss function 
                    such as outputs to regress upon, censoring times for survival analysis, etc...
        fitnessFunc = a python function whose input is a single tuple.  the tuple passed
                    to this function will look like (i,X,otherData[0],otherdata[1], ...) 
                    where i is an index used for sorting the returned fitness score.
                    The output should be a tuple that looks like (i, fitnessScore).
                    Your classification/regression model should be implemented here, but
                    make sure to set any n_cores argument to 1 within this model.
        varNames = a numpy array containing the feature names as strings
        popSize = the number of genotypes per genetion of the population.  larger sizes give 
                    more accurate results but increase the computational burden
        verbose = boolean describing whether statistics should be printed each generation
        n_cores = the number of cores the program can use to parallelize the search
        population = either None, ndarray, or a string.  If None, the population is initialized 
                    uniformly at random.  If a ndarray, this should be a previously found 
                    population to resume the search.  If a string, the folder argument must
                    be provided, and the string must specify the name of the pickled ndarray 
                    (without extension) within the folder from where to continue searching.
        folder = a folder in where a population can be saved or loaded from
        """
        self.fitnessFunc = fitnessFunc
        self.X = X
        self.otherData = otherData if type(otherData) is tuple else (otherData,)
        self.varNames = varNames
        self.numFeats = len(varNames)
        
        self.lr = lr
        self.tol = tol
        self.popSize = popSize
        self.verbose = verbose
        self.generation = 0
        self.scoreRecords = defaultdict(list)
        
        ## init with uniform or given probability
        if probabilities is None:
            self.probabilities = np.ones((1,self.numFeats))*init_prob
        else:
            self.probabilities = self.fixShape(probabilities)
            
        ## setup multicore stuff
        assert type(n_cores) is int and n_cores >= 1
        self.n_cores = n_cores
            
    def fixShape(self, probabilities):
        popErrMsg = "Provided population is not a numpy array"
        shapeErrMsg = "Provided population is wrong length"
        assert type(probabilities) is np.ndarray, popErrMsg
        if probabilities.shape[-1] == 1: # correct shape (n,1)
            probabilities = probabilities[:,0]
        assert probabilities.shape[-1] == self.numFeats, shapeErrMsg
        if len(probabilities.shape) == 1: # correct shape (n,)
            probabilities = probabilities[None,:]
        return probabilities
    
    
    def newGeneration(self):
        # samples new population from probabilities
        population = np.random.random((self.popSize, self.numFeats)) < self.probabilities
        nullIndividuals = ~np.any(population, axis = 1)
        while np.any(nullIndividuals): # fix any members with all Falses by resampling
            popReplace = np.random.random((np.sum(nullIndividuals),self.numFeats)) < self.probabilities
            nullIndividuals[nullIndividuals] = ~np.any(popReplace, axis = 1)
            population[nullIndividuals,:] = popReplace
        return population
        
    
    def evaluate(self, population):
        # loops through each individual and evaluates fitness score
        scores = np.empty(self.popSize)
        generator = ((i, self.X[:,population[i]],
                      *self.otherData) for i in range(self.popSize))
        if self.n_cores == 1:
            for i, x in enumerate(generator):
                i, score = self.fitnessFunc(x)
                scores[i] = score
        else:
            assert self.pool is not None # make sure pool is open
            for i, score in self.pool.map(self.fitnessFunc, generator, chunksize = 1):
                scores[i] = score
        return scores
    
    def updateDist(self, population, scores):
        ## updates probability vector using results of current generation
        
        # sort the population and fitness scores by their performance
        sorter = np.argsort(scores)[::-1]
        scores = scores[sorter]
        population = population[sorter]
        
        # truncation selection
        cut = int(self.popSize/3)
        mod = np.mean(population[:cut,:], axis = 0)
        
        # update using moving average and clip according to tolerance
        self.probabilities = (1-self.lr)*self.probabilities + self.lr*mod[None,:]
        self.probabilities = np.clip(self.probabilities, self.tol, 1-self.tol)
        
    def updateRecords(self, population, scores):
        for i in range(self.popSize):
            record = self.scoreRecords[tuple(population[i])]
            record.append(scores[i])
            ucb_score = statistics.mean(record)
            #if len(record) > 1:
            #    ucb_score += statistics.stdev(record)/math.sqrt(len(record))
            scores[i] = ucb_score
        return scores
            
    def evolve(self, iters = 1, checkpointAt = None, freq = 5):
        ## MAIN FUNCTION USER CALLS
        if self.n_cores > 1:
            self.pool = multiprocessing.Pool(self.n_cores)   
            
        # loop through generations
        for i in range(iters):
            population = self.newGeneration() # sample new population from distribution
            scores = self.evaluate(population) # loop through population and evaluate fitness scores
            ucb_scores = self.updateRecords(population, scores)
            self.updateDist(population, ucb_scores) # update underlying distribution
            self.generation += 1
            if checkpointAt is not None:
                if (i+1) % freq == 0:
                    self.save(checkpointAt)
            if self.verbose:
                print('Fitness calculations finished for generation: ', self.generation)
                print('Max: ', np.round(np.max(scores), 4),
                      '  |  Mean: ', np.round(np.mean(scores), 4),
                      '  |  Min: ', np.round(np.min(scores), 4))
            
        if self.n_cores > 1:
            self.pool.close()
    

        
    def saveProbs(self, loc):
        with open(loc + '_probs.pickle', 'wb') as output:
            pickle.dump(self.probabilities, output, -1)
    def saveRecords(self, loc):
        with open(loc + '_records.pickle', 'wb') as output:
            pickle.dump(self.scoreRecords, output, -1)
    def save(self, loc):
        self.saveProbs(loc)
        self.saveRecords(loc)
            
    def loadProbs(self, loc):
        with open(loc + '_probs.pickle', 'rb') as input:
            self.probabilities = self.fixShape(pickle.load(input))
    def loadRecords(self, loc):
        with open(loc + '_records.pickle', 'rb') as input:
            self.scoreRecords = pickle.load(input)
    def load(self, loc):
        self.loadProbs(loc)
        self.loadRecords(loc)
        
        
        


class PBIL_chooseK(object):
    def __init__(self, fitnessFunc, X, otherData, varNames, K,
                 popSize = 100, lr = 0.5, tol = 0.05, init_prob = 0.5,
                 n_cores = 1, probabilities = None, verbose = True):
        """
        X = numpy ndarray containing the all the input feature data
        otherData = a tuple containing any other relevant data to your fitnesss function 
                    such as outputs to regress upon, censoring times for survival analysis, etc...
        fitnessFunc = a python function whose input is a single tuple.  the tuple passed
                    to this function will look like (i,X,otherData[0],otherdata[1], ...) 
                    where i is an index used for sorting the returned fitness score.
                    The output should be a tuple that looks like (i, fitnessScore).
                    Your classification/regression model should be implemented here, but
                    make sure to set any n_cores argument to 1 within this model.
        varNames = a numpy array containing the feature names as strings
        popSize = the number of genotypes per genetion of the population.  larger sizes give 
                    more accurate results but increase the computational burden
        verbose = boolean describing whether statistics should be printed each generation
        n_cores = the number of cores the program can use to parallelize the search
        population = either None, ndarray, or a string.  If None, the population is initialized 
                    uniformly at random.  If a ndarray, this should be a previously found 
                    population to resume the search.  If a string, the folder argument must
                    be provided, and the string must specify the name of the pickled ndarray 
                    (without extension) within the folder from where to continue searching.
        folder = a folder in where a population can be saved or loaded from
        """
        self.fitnessFunc = fitnessFunc
        self.X = X
        self.otherData = otherData if type(otherData) is tuple else (otherData,)
        self.varNames = varNames
        self.numFeats = K
        
        self.lr = lr
        self.tol = tol
        self.popSize = popSize
        self.verbose = verbose
        self.generation = 0
        self.scoreRecords = defaultdict(list)
        
        ## init with uniform or given probability
        if probabilities is None:
          self.z = np.zeros(len(varNames))
          self.z = self.to_zspace(self.z)
        else:
          self.z = sci.logit(self.fixShape(probabilities))
          
        ## setup multicore stuff
        assert type(n_cores) is int and n_cores >= 1
        self.n_cores = n_cores
        
    def to_zspace(self, arr):
      arr_softmax = sci.softmax(arr)
      arr_inverse_sigmoid  = sci.logit(arr_softmax)
      return np.clip(arr_inverse_sigmoid, -10, 10)


    def update_z(self, z, new_z):
      z += self.lr*(new_z - z)
      z = self.to_zspace(z)
      return z

    def g_to_zspace(self, arr):
      norm_arr = arr/arr.sum()
      inverse_sigmoid = sci.logit(norm_arr)
      return np.clip(inverse_sigmoid, -10, 10)
    
    def fixShape(self, probabilities):
        popErrMsg = "Provided population is not a numpy array"
        shapeErrMsg = "Provided population is wrong length"
        assert type(probabilities) is np.ndarray, popErrMsg
        if probabilities.shape[-1] == 1: # fix shape (n,1)
            probabilities = probabilities[:,0]
        if probabilities.shape[0] == 1: # fix shape (1,n)
            probabilities = probabilities[0,:]
            assert len(probabilities) == self.numFeats, shapeErrMsg
        return probabilities
    
    
    def newGeneration(self):
        # samples new population from probabilities
        population = np.empty((self.popSize, self.numFeats), dtype = np.uint16)
        for i in range(self.popSize):
            population[i] = np.sort(np.random.choice(np.arange(len(self.varNames)), replace = False,
                                  size = self.numFeats, p = self.probabilities))
        return population
        
    
    def evaluate(self, population):
        # loops through each individual and evaluates fitness score
        scores = np.empty(self.popSize)
        generator = ((i, self.X[:,population[i]],
                      *self.otherData) for i in range(self.popSize))
        if self.n_cores == 1:
            for i, x in enumerate(generator):
                i, score = self.fitnessFunc(x)
                scores[i] = score
        else:
            assert self.pool is not None # make sure pool is open
            for i, score in self.pool.map(self.fitnessFunc, generator, chunksize = 1):
                scores[i] = score
        return scores
    
    def updateDist(self, population, scores):
        ## updates probability vector using results of current generation
        
        # sort the population and fitness scores by their performance
        sorter = np.argsort(scores)[::-1]
        scores = scores[sorter]
        population = population[sorter]
        
        # truncation selection
        cut = int(self.popSize/3)
        selected = population[:cut,:]
        mod = np.bincount(selected.flatten(), minlength = len(self.varNames))
        mod = (mod+1)/(np.sum(mod) + len(self.varNames)) # laplace smoothing
        
        # update using moving average and clip according to tolerance
        mod = self.g_to_zspace(mod)
        self.z = self.update_z(self.z, mod)
        
        
        
    def updateRecords(self, population, scores):
        for i in range(self.popSize):
            record = self.scoreRecords[tuple(population[i])]
            record.append(scores[i])
            ucb_score = statistics.mean(record)
            #if len(record) > 1:
            #    ucb_score += statistics.stdev(record)/math.sqrt(len(record))
            scores[i] = ucb_score
        return scores
            
    def evolve(self, iters = 1, checkpointAt = None, freq = 5):
        ## MAIN FUNCTION USER CALLS
        if self.n_cores > 1:
            self.pool = multiprocessing.Pool(self.n_cores)   
            
        # loop through generations
        for i in range(iters):
            population = self.newGeneration() # sample new population from distribution
            scores = self.evaluate(population) # loop through population and evaluate fitness scores
            ucb_scores = self.updateRecords(population, scores)
            self.updateDist(population, ucb_scores) # update underlying distribution
            self.generation += 1
            if checkpointAt is not None:
                if (i+1) % freq == 0:
                    self.save(checkpointAt)
            if self.verbose:
                print('Fitness calculations finished for generation: ', self.generation)
                print('Max: ', np.round(np.max(scores), 4),
                      '  |  Mean: ', np.round(np.mean(scores), 4),
                      '  |  Min: ', np.round(np.min(scores), 4))
            
        if self.n_cores > 1:
            self.pool.close()
    

        
    def saveProbs(self, loc):
        with open(loc + '_probs.pickle', 'wb') as output:
            pickle.dump(self.probabilities, output, -1)
    def saveRecords(self, loc):
        with open(loc + '_records.pickle', 'wb') as output:
            pickle.dump(self.scoreRecords, output, -1)
    def save(self, loc):
        self.saveProbs(loc)
        self.saveRecords(loc)
            
    def loadProbs(self, loc):
        with open(loc + '_probs.pickle', 'rb') as input:
            self.probabilities = self.fixShape(pickle.load(input))
    def loadRecords(self, loc):
        with open(loc + '_records.pickle', 'rb') as input:
            self.scoreRecords = pickle.load(input)
    def load(self, loc):
        self.loadProbs(loc)
        self.loadRecords(loc)