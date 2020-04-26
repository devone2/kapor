import glob

import numpy as np
import cv2 as cv
import utils as cp
import os
import math
from sklearn import neighbors, datasets, svm, preprocessing

files = np.array(glob.glob("labeled/*.jpg"))
#files = ['labeled/xvmdk.jpg']

splitIndex = math.floor(0.8*len(files));
#np.random.seed(0)
np.random.shuffle(files);


training, test = files[:splitIndex ], files[splitIndex:]

Xtrain,Ytrain, trainFileMask = cp.loadLabeledData(training)
Xtest,Ytest, testFileMask = cp.loadLabeledData(test)

print("Train skipped: " + str(np.sum(np.logical_not(trainFileMask))))
print("Test skipped: " + str(np.sum(np.logical_not(testFileMask))))

def showResults(modelName, Z, showFailed = False):
    validCount = np.sum(Z==Ytest)

    print("======= Model: " + modelName)
    print("Test error: " + str(1 - (validCount / len(Z))))
    print("Erorr abs: " + str(len(Z)- validCount) + " / " + str(len(Z)))
    letters = set([chr(chNum) for chNum in list(range(ord('a'),ord('z')+1))])
    digits = set([str(i) for i in list(range(0,10))])
    all = letters.union(digits)

    usedLetters = set(np.unique(Ytrain))

    missingLetters = list(all.difference(usedLetters))
    missingLetters.sort()
    #print(missingLetters)

    # Now find failed and display them:
    mask = np.logical_not(Z==Ytest)

    XtestF = Xtest[mask]
    YtestF = Ytest[mask]
    ZF = Z[mask]
    testF = np.repeat(test[testFileMask], 5)[mask]

    if (showFailed):
        for i in range(0, len(ZF)):
            print(str(i) + ". Expected: " + str(YtestF[i]) + " got: " + str(ZF[i]))
            print("File: " + testF[i])
            img = cp.clearImage(cv.imread(testF[i],0))
            cv.imshow('letters', XtestF[i].reshape((35,30)))
            cv.imshow('image1',img);
            wait = True
            while(wait):
                wait = cv.waitKey(100) != ord('a')
    print("===========")


# KNN part
print("X = " + str(Xtrain.shape));
print("Y = " + str(Ytrain.shape));

def knnModel():
    n_neighbors = 1

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    clf.fit(Xtrain, Ytrain)

    Z = clf.predict(Xtest)
    return Z

def linSVMModel():
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    XtrainS = Xtrain.copy()
    XtestS = Xtest.copy()

    scaler.transform(XtrainS)
    scaler.transform(XtestS)

    clf = svm.LinearSVC()
    clf.fit(XtrainS, Ytrain)
    print("SVM coef:")
    c = clf.coef_
    print(c.shape)

    Z = clf.predict(XtestS)
    return Z



showResults("knn", knnModel())
showResults("linSVM", linSVMModel())

# 
