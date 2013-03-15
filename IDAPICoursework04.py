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
    return mean(theData, axis=0)



def Covariance(theData):
    """ calculates the covariance matrix of a data set represented as in Mean """
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here
    zeroMeanData = realData - Mean(theData)
    covarianceMatrix = dot(transpose(zeroMeanData), zeroMeanData)/ (len(realData)-1)
    # Coursework 4 task 2 ends here
    return covar

def CreateEigenfaceFiles(theBasis, prefix=''):
    for i in range(0,len(theBasis)):
        fileName = prefix + "PrincipalComponent" + str(i) + ".jpg"
        SaveEigenface(theBasis[i], fileName)

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here
    faceImageData = ReadOneImage(theFaceImage)
    magnitudes = dot((faceImageData - theMean), transpose(theBasis))
    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags, prefix=''):
    # Coursework 4 task 5 begins here
    SaveEigenface(aMean, "Reconstructed_0" + ".jpg")
    for i in range(0, len(componentMags)):
        reconstruction = add(dot(transpose(aBasis[0:i]), componentMags[0:i]), aMean)
        SaveEigenface(reconstruction, prefix+"Reconstructed_"+str(i+1)+".jpg")
    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    mean_centered = matrix(theData.astype(float) - Mean(theData))
    UUT = mean_centered * mean_centered.transpose()
    eigenvalues, eigenvectors = linalg.eig(UUT)
    phi = mean_centered.transpose() * eigenvectors
    ## normalising
    phi /= apply_along_axis(linalg.norm, 0, phi)
    indices = argsort(eigenvalues)[::-1]
    orthoPhi = phi.transpose()[indices]
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#

noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
mean_matrix = Mean(theData)
covariance = Covariance(theData)
basis = ReadEigenfaceBasis()
CreateEigenfaceFiles(basis)
meanimage = array(ReadOneImage("MeanImage.jpg"))
projected = ProjectFace(basis, meanimage, "c.pgm")
CreatePartialReconstructions(basis, meanimage, projected)

# 4.6
# pc = PrincipalComponents(theData)
image_data = array(ReadImages())
new_basis = PrincipalComponents(image_data)
CreateEigenfaceFiles(new_basis, "new")
new_mean = Mean(image_data)
new_projected = ProjectFace(new_basis, new_mean, "c.pgm")
CreatePartialReconstructions(new_basis, new_mean, new_projected, "new")
# AppendString("IDAPIResults04.txt","Coursework Four by Mohammad Mirza (mum09) and Oyetola Oyeleye (oo2009)" )


