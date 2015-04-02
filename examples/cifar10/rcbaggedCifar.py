# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Apr 19 2014
"""
A script to run the random and convolution classifcation
"""
import sys
import numpy as np
from time import time

from sklearn.svm import LinearSVC

from CoordinatorFactory import Const, coordinatorRandConvFactory
from Classifier import UnsupervisedVisualBagClassifier as uClassifier
from SubWindowExtractor import SubWindowExtractor
from FilterGenerator import FilterGenerator
from CifarLoader import CifarFromNumpies
from ImageBuffer import FileImageBuffer, NumpyImageLoader
from Logger import formatDuration

#======HISTOGRAM=========#
saveFile = "hist_"
shouldSave = True
#======HYPER PARAMETERS======#
#----RandConv param
#Filtering
nb_filters = 38
filterPolicy = (Const.FGEN_CUSTOM, {"normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_ZEROPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_ZEROPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_IDPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU})
#filterPolicy = (Const.FGEN_IDPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS, "outRange":0.05})
#filterPolicy = (Const.FGEN_IDDIST, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "maxDist":5})
#filterPolicy = (Const.FGEN_STRAT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":0, "maxPerturbation":1})
#
#filterPolicy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_NONE})
#filterPolicy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU})
#filterPolicy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS, "outRange":0.05})
#filterPolicy = (Const.FGEN_IDDIST, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "maxDist":5})
#filterPolicy = (Const.FGEN_STRAT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":minPerturbation, "maxPerturbation":maxPerturbation})

#Aggregation
poolings = [
        (3, 3, Const.POOLING_NONE),
#       (3, 3, Const.POOLING_CONV_AVG),
#       (3, 3, Const.POOLING_CONV_MIN),
#       (3, 3, Const.POOLING_CONV_MAX),
#       (5, 5, Const.POOLING_CONV_AVG),
#       (2, 2, Const.POOLING_AGGREG_AVG),
        ]


#Subwindow
nbSubwindows = 20
subwindowMinSizeRatio = 0.75
subwindowMaxSizeRatio = 1.
subwindowTargetWidth = 16
subwindowTargetHeight = 16
fixedSize = False
subwindowInterpolation = SubWindowExtractor.INTERPOLATION_NEAREST

#Misc.
includeOriginalImage = True
random = False
nbJobs = -1
verbosity = 40
#tempFolder = "/dev/shm"
tempFolder = "/home/jmbegon/jmbegon/code/work/tmp"

#-----BagOfWords params + some SVC params
nbTrees = 750
maxDepth = 30
minSamplesSplit = 500
minSamplesLeaf = 2
randomClassif = True
nbJobsEstimator = -1
verbose = 8
#=====DATA=====#
maxLearningSize = 50000
maxTestingSize = 10000

learningUse = 50000
learningSetDir = "learn/"
learningIndexFile = "0index"

testingUse = 10000
testingSetDir = "test/"
testingIndexFile = "0index"


def formatBigNumber(num):
    revnum = str(num)[::-1]
    right = revnum
    rtn = ""
    for p in range((len(revnum)-1)//3):
        rtn += right[:3]+","
        right = right[3:]
    rtn += right
    return rtn[::-1]

def run(nb_filters=nb_filters,
        filterPolicy=filterPolicy,
        poolings=poolings,
        nbSubwindows=nbSubwindows,
        subwindowMinSizeRatio=subwindowMinSizeRatio,
        subwindowMaxSizeRatio=subwindowMaxSizeRatio,
        subwindowTargetWidth=subwindowTargetWidth,
        subwindowTargetHeight=subwindowTargetHeight,
        fixedSize=fixedSize,
        subwindowInterpolation=subwindowInterpolation,
        includeOriginalImage=includeOriginalImage,
        random=random,
        nbJobs=nbJobs,
        verbosity=verbosity,
        tempFolder=tempFolder,
        nbTrees=nbTrees,
        maxDepth=maxDepth,
        minSamplesSplit=minSamplesSplit,
        minSamplesLeaf=minSamplesLeaf,
        randomClassif=randomClassif,
        nbJobsEstimator=nbJobsEstimator,
        verbose=verbose,
        learningUse=learningUse,
        testingUse=testingUse,
        saveFile=saveFile,
        shouldSave=shouldSave):

    randomState = None
    if not randomClassif:
        randomState = 100

    lsSize = learningUse
    if learningUse > maxLearningSize:
        lsSize = maxLearningSize

    tsSize = testingUse
    if testingUse > maxTestingSize:
        tsSize = maxTestingSize

    #======INSTANTIATING========#
    #--randconv--
    randConvCoord = coordinatorRandConvFactory(
        nbFilters=nb_filters,
        filterPolicy=filterPolicy,
        nbSubwindows=nbSubwindows,
        subwindowMinSizeRatio=subwindowMinSizeRatio,
        subwindowMaxSizeRatio=subwindowMaxSizeRatio,
        subwindowTargetWidth=subwindowTargetWidth,
        subwindowTargetHeight=subwindowTargetHeight,
        poolings=poolings,
        subwindowInterpolation=subwindowInterpolation,
        includeOriginalImage=includeOriginalImage,
        nbJobs=nbJobs,
        verbosity=verbosity,
        tempFolder=tempFolder,
        random=random)

    nb_filters = len(randConvCoord.getFilters())

    #--SVM--
    baseClassif = LinearSVC(verbose=verbose, random_state=randomState)

    #--Classifier
    classifier = uClassifier(coordinator=randConvCoord,
                             base_classifier=baseClassif,
                             n_estimators=nbTrees,
                             max_depth=maxDepth,
                             min_samples_split=minSamplesSplit,
                             min_samples_leaf=minSamplesLeaf,
                             n_jobs=nbJobsEstimator,
                             random_state=randomState,
                             verbose=verbose)

    #--Data--
    loader = CifarFromNumpies(learningSetDir, learningIndexFile)
    learningSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    learningSet = learningSet[0:lsSize]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:tsSize]

    #=====COMPUTATION=====#
    #--Learning--#
    print "Starting learning"
    fitStart = time()
    hist = classifier._preprocess(learningSet, learningPhase=True)
    y = learningSet.getLabels()
    if shouldSave:
        np.savez(saveFile, data=hist.data, indices=hist.indices,
                 indptr=hist.indptr, shape=hist.shape)
    classifier.fit_histogram(hist, y)
    fitEnd = time()
    print "Learning done", formatDuration(fitEnd-fitStart)
    sys.stdout.flush()

    #--Testing--#
    y_truth = testingSet.getLabels()
    predStart = time()
    y_pred = classifier.predict(testingSet)
    predEnd = time()
    accuracy = classifier.accuracy(y_pred, y_truth)
    confMat = classifier.confusionMatrix(y_pred, y_truth)

    #====ANALYSIS=====#
    importance, order = randConvCoord.importancePerFeatureGrp(classifier._visualBagger)

    print "==================Bag of Visual Words======================="
    print "-----------Filtering--------------"
    print "nb_filters", nb_filters
    print "filterPolicy", filterPolicy
    print "----------Pooling--------------"
    print "poolings", poolings
    print "--------SW extractor----------"
    print "#Subwindows", nbSubwindows
    print "subwindowMinSizeRatio", subwindowMinSizeRatio
    print "subwindowMaxSizeRatio", subwindowMaxSizeRatio
    print "subwindowTargetWidth", subwindowTargetWidth
    print "subwindowTargetHeight", subwindowTargetHeight
    print "fixedSize", fixedSize
    print "------------Misc-----------------"
    print "includeOriginalImage", includeOriginalImage
    print "random", random
    print "tempFolder", tempFolder
    print "verbosity", verbosity
    print "nbJobs", nbJobs
    print "--------Bag of words params + SVC----------"
    print "nbTrees", nbTrees
    print "maxDepth", maxDepth
    print "minSamplesSplit", minSamplesSplit
    print "minSamplesLeaf", minSamplesLeaf
    print "nbJobsEstimator", nbJobsEstimator
    print "verbose", verbose
    print "randomState", randomState
    print "------------Data---------------"
    print "LearningSet size", len(learningSet)
    print "TestingSet size", len(testingSet)
    print "-------------------------------"
    if shouldSave:
        print "saveFile", saveFile
    print "Fit time", formatDuration(fitEnd-fitStart)
    print "Classifcation time", formatDuration(predEnd-predStart)
    print "Accuracy", accuracy
    print "Leafs", formatBigNumber(classifier.histoSize)

    return accuracy, confMat, importance, order

if __name__ == "__main__":
    acc, confMat, importance, order = run()

    print "Confusion matrix :\n", confMat
    print "Feature importance :\n", importance
    print "Feature importance order :\n", order
