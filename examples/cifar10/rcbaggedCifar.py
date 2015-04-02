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

from randconv import randconv_factory, Const
from randconv import UnsupervisedVisualBagClassifier as uClassifier
from randconv.image import (SubWindowExtractor, FilterGenerator,
                            FileImageBuffer, NumpyImageLoader)
from progressmonitor import format_duration

from .cifardb import *


#======HISTOGRAM=========#
save_file = "hist_"
should_save = True
#======HYPER PARAMETERS======#
#----RandConv param
#Filtering
nb_filters = 38
filter_policy = (Const.FGEN_CUSTOM, {"normalization":FilterGenerator.NORMALISATION_NONE})
#filter_policy = (Const.FGEN_ZEROPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_NONE})
#filter_policy = (Const.FGEN_ZEROPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_NONE})
#filter_policy = (Const.FGEN_IDPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU})
#filter_policy = (Const.FGEN_IDPERT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS, "outRange":0.05})
#filter_policy = (Const.FGEN_IDDIST, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "maxDist":5})
#filter_policy = (Const.FGEN_STRAT, {"minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":0, "maxPerturbation":1})
#
#filter_policy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_NONE})
#filter_policy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_NONE})
#filter_policy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU})
#filter_policy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS, "outRange":0.05})
#filter_policy = (Const.FGEN_IDDIST, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_RU, "maxDist":5})
#filter_policy = (Const.FGEN_STRAT, {"sparseProb":0.25, "minSize":2, "maxSize":32, "minVal":-1, "maxVal":1, "valGen":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":minPerturbation, "maxPerturbation":maxPerturbation})

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
nb_subwindows = 20
sw_min_size_ratio = 0.75
sw_max_size_ratio = 1.
sw_target_width = 16
sw_target_height = 16
fixed_size = False
sw_interpolation = SubWindowExtractor.INTERPOLATION_NEAREST

#Misc.
include_original_img = True
random = False
n_jobs = -1
verbosity = 40
#temp_folder = "/dev/shm"
temp_folder = "/home/jmbegon/jmbegon/code/work/tmp"

#-----BagOfWords params + some SVC params
nb_trees = 750
max_depth = 30
min_samples_split = 500
min_samples_leaf = 2
random_classif = True
n_jobsEstimator = -1
verbose = 8
#=====DATA=====#
maxLearningSize = 50000
maxTestingSize = 10000

learning_use = 50000
learningSetDir = "learn/"
learningIndexFile = "0index"

testing_use = 10000
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
        filter_policy=filter_policy,
        poolings=poolings,
        nb_subwindows=nb_subwindows,
        sw_min_size_ratio=sw_min_size_ratio,
        sw_max_size_ratio=sw_max_size_ratio,
        sw_target_width=sw_target_width,
        sw_target_height=sw_target_height,
        fixed_size=fixed_size,
        sw_interpolation=sw_interpolation,
        include_original_img=include_original_img,
        random=random,
        n_jobs=n_jobs,
        verbosity=verbosity,
        temp_folder=temp_folder,
        nb_trees=nb_trees,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_classif=random_classif,
        n_jobsEstimator=n_jobsEstimator,
        verbose=verbose,
        learning_use=learning_use,
        testing_use=testing_use,
        save_file=save_file,
        should_save=should_save):

    rand_state = None
    if not random_classif:
        rand_state = 100

    ls_size = learning_use
    if learning_use > maxLearningSize:
        ls_size = maxLearningSize

    ts_size = testing_use
    if testing_use > maxTestingSize:
        ts_size = maxTestingSize

    #======INSTANTIATING========#
    #--randconv--
    randConvCoord = randconv_factory(
        nb_filters=nb_filters,
        filter_policy=filter_policy,
        nb_subwindows=nb_subwindows,
        sw_min_size_ratio=sw_min_size_ratio,
        sw_max_size_ratio=sw_max_size_ratio,
        sw_target_width=sw_target_width,
        sw_target_height=sw_target_height,
        poolings=poolings,
        sw_interpolation=sw_interpolation,
        include_original_img=include_original_img,
        n_jobs=n_jobs,
        verbosity=verbosity,
        temp_folder=temp_folder,
        random=random)

    nb_filters = len(randConvCoord.get_filters())

    #--SVM--
    baseClassif = LinearSVC(verbose=verbose, random_state=rand_state)

    #--Classifier
    classifier = uClassifier(coordinator=randConvCoord,
                             base_classifier=baseClassif,
                             n_estimators=nb_trees,
                             max_depth=max_depth,
                             min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf,
                             n_jobs=n_jobsEstimator,
                             random_state=rand_state,
                             verbose=verbose)

    #--Data--
    loader = CifarFromNumpies(learningSetDir, learningIndexFile)
    learningSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    learningSet = learningSet[0:ls_size]

    loader = CifarFromNumpies(testingSetDir, testingIndexFile)
    testingSet = FileImageBuffer(loader.getFiles(), NumpyImageLoader())
    testingSet = testingSet[0:ts_size]

    #=====COMPUTATION=====#
    #--Learning--#
    print "Starting learning"
    fitStart = time()
    hist = classifier._preprocess(learningSet, learningPhase=True)
    y = learningSet.getLabels()
    if should_save:
        np.savez(save_file, data=hist.data, indices=hist.indices,
                 indptr=hist.indptr, shape=hist.shape)
    classifier.fit_histogram(hist, y)
    fitEnd = time()
    print "Learning done", format_duration(fitEnd-fitStart)
    sys.stdout.flush()

    #--Testing--#
    y_truth = testingSet.getLabels()
    predStart = time()
    y_pred = classifier.predict(testingSet)
    predEnd = time()
    accuracy = classifier.accuracy(y_pred, y_truth)
    confMat = classifier.confusion_matrix(y_pred, y_truth)

    #====ANALYSIS=====#
    importance, order = randConvCoord.importancePerFeatureGrp(classifier._visualBagger)

    print "==================Bag of Visual Words======================="
    print "-----------Filtering--------------"
    print "nb_filters", nb_filters
    print "filter_policy", filter_policy
    print "----------Pooling--------------"
    print "poolings", poolings
    print "--------SW extractor----------"
    print "#Subwindows", nb_subwindows
    print "sw_min_size_ratio", sw_min_size_ratio
    print "sw_max_size_ratio", sw_max_size_ratio
    print "sw_target_width", sw_target_width
    print "sw_target_height", sw_target_height
    print "fixed_size", fixed_size
    print "------------Misc-----------------"
    print "include_original_img", include_original_img
    print "random", random
    print "temp_folder", temp_folder
    print "verbosity", verbosity
    print "n_jobs", n_jobs
    print "--------Bag of words params + SVC----------"
    print "nb_trees", nb_trees
    print "max_depth", max_depth
    print "min_samples_split", min_samples_split
    print "min_samples_leaf", min_samples_leaf
    print "n_jobsEstimator", n_jobsEstimator
    print "verbose", verbose
    print "rand_state", rand_state
    print "------------Data---------------"
    print "LearningSet size", len(learningSet)
    print "TestingSet size", len(testingSet)
    print "-------------------------------"
    if should_save:
        print "save_file", save_file
    print "Fit time", format_duration(fitEnd-fitStart)
    print "Classifcation time", format_duration(predEnd-predStart)
    print "Accuracy", accuracy
    print "Leafs", formatBigNumber(classifier.histoSize)

    return accuracy, confMat, importance, order

if __name__ == "__main__":
    acc, confMat, importance, order = run()

    print "Confusion matrix :\n", confMat
    print "Feature importance :\n", importance
    print "Feature importance order :\n", order
