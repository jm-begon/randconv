# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 10 2014
"""
A script to run the random and convolution classifcation
"""
import sys
from time import time

from sklearn.ensemble import ExtraTreesClassifier

from randconv import randconv_factory, Classifier, Const
from randconv.image import (SubWindowExtractor, FilterGenerator,
                            FileImageBuffer, NumpyImageLoader)
from progressmonitor import format_duration

from cifardb import *

#======PROB MATRIX=========#
save_file = "rc_"
should_save = True
#======HYPER PARAMETERS======#
#----RandConv param
#Filtering
nb_filters = 100
#filter_policy = (Const.FGEN_ZEROPERT, {"min_size":2, "max_size":10, "min_val":-1, "max_val":1, "value_generator":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_NONE})
filter_policy = (Const.FGEN_ZEROPERT, {"min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_NONE})
#filter_policy = (Const.FGEN_IDPERT, {"min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_RU})
#filter_policy = (Const.FGEN_IDPERT, {"min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_GAUSS, "outRange":0.05})
#filter_policy = (Const.FGEN_IDDIST, {"min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_RU, "maxDist":5})
#filter_policy = (Const.FGEN_STRAT, {"min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":0, "maxPerturbation":1})
#
#filter_policy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_RU, "normalization":FilterGenerator.NORMALISATION_NONE})
#filter_policy = (Const.FGEN_ZEROPERT, {"sparseProb":0.25, "min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_SET, "probLaw":[(-1, 0.3), (0, 0.4), (1, 0.3)], "normalization":FilterGenerator.NORMALISATION_NONE})
#filter_policy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_RU})
#filter_policy = (Const.FGEN_IDPERT, {"sparseProb":0.25, "min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_GAUSS, "outRange":0.05})
#filter_policy = (Const.FGEN_IDDIST, {"sparseProb":0.25, "min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_RU, "maxDist":5})
#filter_policy = (Const.FGEN_STRAT, {"sparseProb":0.25, "min_size":2, "max_size":32, "min_val":-1, "max_val":1, "value_generator":Const.RND_GAUSS,  "outRange":0.001, "strat_nbCells":10, "minPerturbation":0, "maxPerturbation":1})

#Aggregation
poolings = [
        (2, 2, Const.POOLING_AGGREG_AVG),
#        (3, 3, Const.POOLING_MW_AVG),
#        (3, 3, Const.POOLING_MW_MIN),
#        (3, 3, Const.POOLING_MW_MAX)
        ]

#Extraction
extractor = (Const.FEATEXT_ALL, {})
#extractor =  (Const.FEATEXT_SPASUB, {"nbCol":2})
#extractor =  (Const.FEATEXT_SPASUB, {"nbCol":1})

#Subwindow
nb_subwindows = 10
sw_min_size_ratio = 0.75
sw_max_size_ratio = 1.
sw_target_width = 16
sw_target_height = 16
fixedSize = False
sw_interpolation = SubWindowExtractor.INTERPOLATION_NEAREST

#Misc.
include_original_img = True
random = False
n_jobs = -1
verbosity = 40
#temp_folder = "/dev/shm"
temp_folder = "/home/jmbegon/jmbegon/code/work/tmp"

#-----Extratree param
nbTrees = 30
maxFeatures = "auto"
maxDepth = None
minSamplesSplit = 2
minSamplesLeaf = 1
bootstrap = False
random_classif = True
n_jobsEstimator = -1
verbose = 8

#=====DATA=====#
maxLearningSize = 50000
maxTestingSize = 10000

learning_use = 500
learningSetDir = "learn/"
learningIndexFile = "0index"

testing_use = 100
testingSetDir = "test/"
testingIndexFile = "0index"


def run(nb_filters=nb_filters,
        filter_policy=filter_policy,
        poolings=poolings,
        extractor=extractor,
        nb_subwindows=nb_subwindows,
        sw_min_size_ratio=sw_min_size_ratio,
        sw_max_size_ratio=sw_max_size_ratio,
        sw_target_width=sw_target_width,
        sw_target_height=sw_target_height,
        fixedSize=fixedSize,
        sw_interpolation=sw_interpolation,
        include_original_img=include_original_img,
        random=random,
        n_jobs=n_jobs,
        verbosity=verbosity,
        temp_folder=temp_folder,
        nbTrees=nbTrees,
        maxFeatures=maxFeatures,
        maxDepth=maxDepth,
        minSamplesSplit=minSamplesSplit,
        minSamplesLeaf=minSamplesLeaf,
        bootstrap=bootstrap,
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
    #--RandConv--
    randConvCoord = randconv_factory(
        nb_filters=nb_filters,
        filter_policy=filter_policy,
        poolings=poolings,
        extractor=extractor,
        nb_subwindows=nb_subwindows,
        sw_min_size_ratio=sw_min_size_ratio,
        sw_max_size_ratio=sw_max_size_ratio,
        sw_target_width=sw_target_width,
        sw_target_height=sw_target_height,
        sw_interpolation=sw_interpolation,
        include_original_img=include_original_img,
        n_jobs=n_jobs,
        verbosity=verbosity,
        temp_folder=temp_folder,
        random=random)

    #--Extra-tree--
    baseClassif = ExtraTreesClassifier(nbTrees,
                                       max_features=maxFeatures,
                                       max_depth=maxDepth,
                                       min_samples_split=minSamplesSplit,
                                       min_samples_leaf=minSamplesLeaf,
                                       bootstrap=bootstrap,
                                       n_jobs=n_jobsEstimator,
                                       random_state=rand_state,
                                       verbose=verbose)

     #--Classifier
    classifier = Classifier(randConvCoord, baseClassif)

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
    classifier.fit(learningSet)
    fitEnd = time()
    print "Learning done", format_duration(fitEnd-fitStart)
    sys.stdout.flush()

    #--Testing--#
    y_truth = testingSet.getLabels()
    predStart = time()
    y_prob, y_pred = classifier.predict_predict_proba(testingSet)
    predEnd = time()
    accuracy = classifier.accuracy(y_pred, y_truth)
    confMat = classifier.confusion_matrix(y_pred, y_truth)

    #====ANALYSIS=====#
    importance, order = randConvCoord.importance_per_feature_grp(baseClassif)

    print "==================RandConv================"
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
    print "fixedSize", fixedSize
    print "------------Misc-----------------"
    print "include_original_img", include_original_img
    print "random", random
    print "temp_folder", temp_folder
    print "verbosity", verbosity
    print "n_jobs", n_jobs
    print "--------ExtraTrees----------"
    print "nbTrees", nbTrees
    print "maxFeatures", maxFeatures
    print "maxDepth", maxDepth
    print "minSamplesSplit", minSamplesSplit
    print "minSamplesLeaf", minSamplesLeaf
    print "bootstrap", bootstrap
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

    if should_save:
        np.save(save_file, y_prob)

    return accuracy, confMat, importance, order

if __name__ == "__main__":
    acc, confMat, importance, order = run()

    print "Confusion matrix :\n", confMat
    print "Feature importance :\n", importance
    print "Feature importance order :\n", order
