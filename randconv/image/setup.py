# -*- coding: utf-8 -*-
"""
setup script
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__version__ = 'dev'


from numpy.distutils.misc_util import Configuration
import os, numpy

def configuration(parent_package="", top_path=None):
    config = Configuration("image", parent_package, top_path)

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    config.add_extension("_fast_pooling",
                         sources=["_fast_pooling.c"],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         extra_compile_args=["-O3", "-Wno-unused-function"])



    config.add_subpackage('test')

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
