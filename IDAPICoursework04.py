#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *

# Coursework 4 begins here
def Mean(theData):
    """ This calculates the mean vector of a data set 
        represented (as usual) by a matrix in which the rows 
        are data points and the columns are variables.
    """
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here
    import pdb; pdb.set_trace()


    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    """ calculates the covariance matrix of a data set represented as in Mean """
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here
    import pdb; pdb.set_trace()

    # Coursework 4 task 2 ends here
    return covar

def CreateEigenfaceFiles(theBasis):
    pass

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    pass
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

noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
mean = Mean(theData)

# AppendString("IDAPIResults04.txt","Coursework Four by Mohammad Mirza (mum09) and Oyetola Oyeleye (oo2009)" )


