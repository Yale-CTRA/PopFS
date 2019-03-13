# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:15:20 2019

@author: adityabiswas
"""


import numpy as np
import multiprocessing
import pickle
import os


class GeneticFS(object):
    def __init__(self, fitnessFunc, X, otherData, varNames, numFeats = 10, popSize = 100, 
                             numClones = 1, mutationRate = 0.01, n_cores = 1, population = None,
                             verbose = True, folder = None):
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
        numFeats = the number of features the algorithm should optimize for
        popSize = the number of genotypes per genetion of the population.  larger sizes give 
                    more accurate results but increase the computational burden
        numClones = number of individuals in each generation retained by elitism
        mutationRate = the probability a feature flips to a random feature from the full set
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
        self.numFeats = numFeats
        self.popSize = popSize
        self.numClones = numClones
        self.mutationRate = mutationRate
        self.generation = 0
        self.n_cores = n_cores
        self.verbose = verbose
        self.folder = folder
        
        ## init population with random feature sets, possibly appending
        ## to a previous population till popSize is met
        randomPopulation = np.array([np.random.choice(np.arange(len(self.varNames)), 
                                    self.numFeats, replace = False) for i in range(popSize)])
        if population is None:
            self.population = randomPopulation
        else:
            if type(population) is str:
                folderErrMsg = "Folder must be provided if population parameter is a string"
                assert folder is not None, folderErrMsg
                population = self.load(population)
            
            popErrMsg = "Provided population contains the wrong number of features"
            assert type(population) is np.ndarray and np.shape(population)[1] == numFeats, popErrMsg
            if len(population) >= popSize:
                self.population = population[:popSize,:]
            else:
                k = popSize - len(population)
                self.population = np.concatenate([population, randomPopulation[:k,:]], axis = 0)
        
        self.clearScores()
        self.probs = self.linearPMF(popSize)
        self.open()
        
    def recombine(self, x0, x1):
        x0, x1 = set(x0), set(x1)
        z = x0 & x1     # find common features
        choices = np.array(list((x0 | x1) - z))     # combine uncommon features
        z = np.array(list(z))
        np.random.shuffle(choices)
        k = self.numFeats - len(z)
        y = np.concatenate((choices[:k], z)) # randomly pick k uncommon features
        return y
    
    def mutate(self, individual):
        # randomly chooses whether to mutate each var
        mutateVar = np.random.uniform(size = self.numFeats) < self.mutationRate
        for i in range(self.numFeats):
            if mutateVar[i]: # if chosen, replace with var not currently in genotype
                choices = np.array(list(set(np.arange(len(self.varNames))) - set(individual)))
                individual[i] = np.random.choice(choices)
        return individual
    
        
    def mate(self, population):
        # samples all mating pairs
        size = self.popSize - self.numClones
        father = np.random.choice(np.arange(self.popSize),
                                    size = size, p = self.probs)
        mother = np.random.choice(np.arange(self.popSize),
                                    size = size, p = self.probs)
        
        # genetic recombination and mutation for all pairs
        children = np.empty((size, self.numFeats), dtype = np.int32)
        for i in range(size):
            child = self.recombine(population[father[i]], 
                                    population[mother[i]])
            children[i,:] = self.mutate(child)
        return children

    def newGeneration(self):
        populationA = self.population[:self.numClones] # clone copies
        populationB = self.mate(self.population) # generate mating offspring
        self.population = np.concatenate((populationA, populationB), axis = 0)
        assert len(self.population) == self.popSize
        
    def evaluate(self):
        self.clearScores()
        assert self.pool is not None
        # train models on each member of the population's training set
        # and evaluate on the test set
        generator = ((i, self.X[:,self.population[i]],
                      *self.otherData) for i in range(self.popSize))
        for i, score in self.pool.map(self.fitnessFunc, generator, chunksize = 1):
            self.scores[i] = score
        
        # sort the population and fitness scores by the performance
        sorter = np.argsort(self.scores)[::-1]
        self.scores = self.scores[sorter]
        self.population = self.population[sorter]
            
    def evolve(self):
        self.newGeneration() # make new gen using last gen fitness scores
        self.evaluate() # generate current fitness scores and sort
        if self.verbose:
            print('Fitness calculations finished for generation: ', self.generation)
            print('Max: ', np.round(np.max(self.scores), 4), 
                  '   |   Min: ', np.round(np.min(self.scores), 4))
        self.generation += 1
    
    def getBest(self, n = 1):
        # returns the best fitness scores in current population 
        # and their associated feature sets
        assert n >= 1
        if n == 1:
            return (self.scores[0], self.varNames[self.population[0]])
        else:
            return [(self.scores[idx], self.varNames[self.population[idx]]) for idx in range(n)]
    
    def linearPMF(self, N):
        # rank-based linearly decreasing pmf (top ranked is x=0)
        return -(2*np.arange(N) - 2*N + 1)/(N**2)

    
    def getPopulation(self):
        return self.population
    def getScores(self):
        return self.scores        
    def getGeneration(self):
        return self.generation
    def setMutationRate(self, rate):
        self.mutationRate = rate
    def setFolder(self, folder):
        self.folder = folder
    def clearScores(self):
        self.scores = np.empty(self.popSize)
    
    def open(self):
        self.pool = multiprocessing.Pool(self.n_cores)     
    def close(self):
        self.pool.close()
        
    def save(self, name):
        name += '.pickle'
        with open(os.path.join(self.folder, name), 'wb') as output:
            pickle.dump(self.population, output, -1)
    
    def load(self, name):
        name += '.pickle'
        with open(os.path.join(self.folder, name), 'rb') as input_:
            return pickle.load(input_)        
        