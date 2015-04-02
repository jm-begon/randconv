# -*- coding: utf-8 -*-
"""
setup script
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__version__ = 'dev'


from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):

    config = Configuration("util", parent_package, top_path)
    
    config.add_subpackage('test')

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
