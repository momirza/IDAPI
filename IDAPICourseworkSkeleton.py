#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
# Coursework 1 task 1 should be inserted here
    prior = array(map(float, [len([x for x in theData if x[0] == y]) for y in xrange(noStates[root])]))/len(theData)
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
# Coursework 1 task 2 should be inserted here
    prob = lambda appearances: [float(len([x for x in appearances if x == y]))/len(appearances) \
                                                    for y in xrange(noStates[varC])]
    cPT = array([prob([x[varC] for x in theData if x[0] == y]) for y in xrange(noStates[varP])]).transpose()
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here 
    prob = lambda appearances: [float(len([x for x in appearances if x == y]))/len(theData) \
                                                    for y in xrange(noStates[varRow])]
    jPT = array([prob([x[varRow] for x in theData if x[0] == y]) for y in xrange(noStates[varCol])]).transpose()
    
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here 
    aCPT = aJPT/sum(aJPT, axis=0)
# coursework 1 taks 4 ends here
    return aCPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    def mult(x,y): return x*y
    rootPdf = naiveBayes[0] * reduce(mult, [naiveBayes[x+1][y] for x,y in enumerate(theQuery)])
    rootPdf = rootPdf / sum(rootPdf)
# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
   

# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    

# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    

# end of coursework 2 task 3
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
  
    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
   

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
AppendString("IDAPIResults01.txt","Coursework One Results by Mohammad Mirza (mum09) and Oyetola Oyeleye (oo2009) ")
AppendString("IDAPIResults01.txt","") #blank line
prior = Prior(theData, 0, noStates)
cpt_1_0 = CPT(theData, 1, 0, noStates)
cpt_2_0 = CPT(theData, 2, 0, noStates)
cpt_3_0 = CPT(theData, 3, 0, noStates)
cpt_4_0 = CPT(theData, 4, 0, noStates)
cpt_5_0 = CPT(theData, 5, 0, noStates)
jpt_2_0 = JPT(theData, 2, 0, noStates)
cptfromjpt = JPT2CPT(jpt_2_0)
q1 = [4, 0, 0, 0, 5]
q2 = [6, 5, 2, 5, 5]
naive_network = [prior, cpt_1_0, cpt_2_0, cpt_3_0, cpt_4_0, cpt_5_0]
posterior_probability1 = Query(q1, naive_network)
posterior_probability2 = Query(q2, naive_network)
AppendString('IDAPIResults01.txt', 'Prior:')
AppendList('IDAPIResults01.txt', prior)
AppendString('IDAPIResults01.txt', 'P(2|0):')
AppendArray('IDAPIResults01.txt', cpt_2_0)
AppendString('IDAPIResults01.txt', 'P(2&0):')
AppendArray('IDAPIResults01.txt', jpt_2_0)
AppendString('IDAPIResults01.txt', 'P(2|0) from P(2&0):')
AppendArray('IDAPIResults01.txt', cptfromjpt)
AppendString('IDAPIResults01.txt', 'Query %s' % q1)
AppendList('IDAPIResults01.txt', posterior_probability1)
AppendString('IDAPIResults01.txt', 'Query %s' % q2)
AppendList('IDAPIResults01.txt', posterior_probability2)


import pdb; pdb.set_trace()
pass

#
# continue as described
#
#


