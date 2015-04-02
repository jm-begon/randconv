# -*- coding: utf-8 -*-
"""
Util Package
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__version__ = 'dev'


from .number_generator import (NumberGenerator, IntegerUniformGenerator,
                               OddUniformGenerator, GaussianNumberGenerator,
                               ClippedGaussianRNG, ConstantGenerator,
                               CustomDiscreteNumberGenerator)
from .numpy_factory import NumpyFactory
from .taskmanager import SerialExecutor, ParallelExecutor

__all__ = ["NumberGenerator", "IntegerUniformGenerator", "OddUniformGenerator",
           "GaussianNumberGenerator", "ClippedGaussianRNG", "ConstantGenerator",
           "CustomDiscreteNumberGenerator", "NumpyFactory", "SerialExecutor",
           "ParallelExecutor"]
