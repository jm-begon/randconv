# -*- coding: utf-8 -*-
"""
Created on Sun May 04 13:02:09 2014

@author: Jm Begon
"""

import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):

    config = Configuration("randconv", parent_package, top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension("_compute_histogram",
                         sources=["_compute_histogram.c"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3", "-Wno-unused-function"])

    config.add_subpackage("image")
    config.add_subpackage("util")
    config.add_subpackage('test')

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())

