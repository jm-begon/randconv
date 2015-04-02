# -*- coding: utf-8 -*-
"""

"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__version__ = 'dev'
__date__ = "20 January 2015"


from ._compute_histogram import compute_histogram
from .classifier import Classifier, UnsupervisedVisualBagClassifier
from .coordinator import PyxitCoordinator, RandConvCoordinator
from .coordinator_factory import Const, pyxit_factory, randconv_factory
from .feature_extractor import ImageLinearizationExtractor, DepthCompressorILE

__all__ = ["compute_histogram", "Classifier", "UnsupervisedVisualBagClassifier",
           "PyxitCoordinator", "RandConvCoordinator", "Const", "pyxit_factory",
           "randconv_factory", "ImageLinearizationExtractor",
           "DepthCompressorILE"]
