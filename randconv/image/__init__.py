# -*- coding: utf-8 -*-
"""
Main package
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__version__ = 'dev'


from ._fast_pooling import FastMWAvgPooler, FastMWMaxPooler, FastMWMinPooler
from .aggregator import AverageAggregator, MaximumAggregator, MinimumAggregator
from .convolutional_extractor import ConvolutionalExtractor
from .convolver import Convolver, RGBConvolver
from .filter_generator import (FilterGenerator, FixSizeFilterGenerator,
                               IdPerturbatedFG, IdMaxL1DistPerturbFG,
                               StratifiedFG, SparsityDecoratorFG, FiniteFilter,
                               Finite3Filter, Finite3SameFilter, OrderedMFF)
from .filter_holder import custom_filters, custom_finite_3_same_filter
from .numpy_pil_converter import (NumpyPILConvertor, NumpyToPILConvertor,
                                  PILToNumpyConvertor)
from .imagebuffer import (ImageBuffer, FileImageBuffer, ImageLoader,
                          NotCMapImageLoader, NumpyImageLoader)
from .pooler import (Pooler, MultiPooler, IdentityPooler, ConvolutionalPooler,
                     ConvMaxPooler, ConvMinPooler, ConvAvgPooler,
                     MorphOpeningPooler, MorphClosingPooler,
                     MorphErosionGradientPooler, MorphDilationGradientPooler,
                     MorphGradientPooler, MorphLaplacianPooler,
                     MorphBlackHatPooler, MorphWhiteHatPooler)
from .subwindow_extractor import (SubWindowExtractor, FixTargetSWExtractor,
                                  FixImgSWExtractor, MultiSWExtractor)
from .rescaler import Rescaler, MaxoutRescaler

__all__ = ["FastMWAvgPooler", "FastMWMaxPooler", "FastMWMinPooler",
           "AverageAggregator", "MaximumAggregator", "MinimumAggregator",
           "ConvolutionalExtractor", "Convolver", "RGBConvolver",
           "FilterGenerator", "FixSizeFilterGenerator", "IdPerturbatedFG",
           "IdMaxL1DistPerturbFG", "StratifiedFG", "SparsityDecoratorFG",
           "FiniteFilter", "Finite3Filter", "Finite3SameFilter",
           "custom_filters", "custom_finite_3_same_filter",
           "NumpyPILConvertor", "NumpyToPILConvertor", "PILToNumpyConvertor",
           "ImageBuffer", "FileImageBuffer", "ImageLoader",
           "NotCMapImageLoader", "NumpyImageLoader", "Pooler", "MultiPooler",
           "IdentityPooler", "ConvolutionalPooler", "ConvMaxPooler",
           "ConvMinPooler", "ConvAvgPooler", "MorphOpeningPooler",
           "MorphClosingPooler", "MorphErosionGradientPooler",
           "MorphDilationGradientPooler", "MorphGradientPooler",
           "MorphLaplacianPooler", "MorphBlackHatPooler", "MorphWhiteHatPooler",
           "SubWindowExtractor", "FixTargetSWExtractor", "FixImgSWExtractor",
           "MultiSWExtractor", "Rescaler", "MaxoutRescaler"]
