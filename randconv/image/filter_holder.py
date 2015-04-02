# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Apr 02 2014

"""
A custom filter container
"""
import numpy as np
from math import sqrt

from randconv.image import Finite3SameFilter

__all__ = ["custom_filters", "custom_finite_3_same_filter"]


def shape2D(squareIterable, normalisation=None):
    """
    Return the corresponding 2D filter

    Parametes
    ---------
    squareIterable : iterable of number whose length is a square interger
        the 1D form of the filter

    normalization : (default : None)
        TODO XXX
    Return
    ------
    filt : 2D numpy array
        The 2D filter

    Example
    -------
    >>> sobelHz = [-1,0,1,-2,0,2,-1,0,1]
    >>> shape2D(sobelHz) # doctest: +SKIP
    array([[-1.,  0.,  1.],
           [-2.,  0.,  2.],
           [-1.,  0.,  1.]])
    """
    size = int(sqrt(len(squareIterable)))
    if size*size != len(squareIterable):
        raise ValueError("The length of the iterable must be a square integer")
    filt = np.zeros((size, size))
    for i, val in enumerate(squareIterable):
        x = i // size
        y = i % size
        filt[x][y] = val
    return filt


def custom_filters():
    filters = []

    centralEmphasis = shape2D([0.075, 0.125, 0.075, 0.125, 0.2, 0.125,
                               0.075, 0.125, 0.075])
    filters.append(centralEmphasis)

    #discrete, two-dimensional gaussian 5x5 (which stdev ?)
    gauss5x5 = shape2D([0, 1, 2, 1, 0, 1, 3, 5, 3, 1, 2, 5, 9, 5, 2, 1, 3,
                        5, 3, 1, 0, 1, 2, 1, 0])
    filters.append(gauss5x5)

    #Derivative
    sobelHz = shape2D([-1, 0, 1, -2, 0, 2, -1, 0, 1])
    filters.append(sobelHz)

    sobelV = shape2D([1, 2, 1, 0, 0, 0, -1, -2, -1])
    filters.append(sobelV)

    laplaceIso = shape2D([0, 1, 0, 1, -4, 1, 0, 1, 0])
    filters.append(laplaceIso)

    laplaceFullIso = shape2D([1, 1, 1, 1, -8, 1, 1, 1, 1])
    filters.append(laplaceFullIso)

    bigLaplace = shape2D([0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1,
                          0, -1, -2, -1, 0, 0, 0, -1, 0, 0])
    filters.append(bigLaplace)

    bigLaplace2 = shape2D([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 24, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1])
    filters.append(bigLaplace2)

    #Inverted
    laplaceIsoInv = shape2D([0, -1, 0, -1, 4, -1, 0, -1, 0])
    filters.append(laplaceIsoInv)

    laplaceFullIsoInv = shape2D([-1, -1, -1, -1, 8, -1, 1, -1, -1])
    filters.append(laplaceFullIsoInv)

    #Prewitt
    prewittHz = shape2D([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    filters.append(prewittHz)

    prewittV = shape2D([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    filters.append(prewittV)

    #Oriented edges
    hzEdges = shape2D([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 2, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0])
    filters.append(hzEdges)

    vEdges = shape2D([0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 4, 0, 0, 0, 0, -1,
                      0, 0, 0, 0, -1, 0, 0])
    filters.append(vEdges)

    plus45Edges = shape2D([-1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 6, 0, 0, 0,
                           0, 0, -2, 0, 0, 0, 0, 0, -1])
    filters.append(plus45Edges)

    minus45Edges = shape2D([0, 0, 0, 0, -1, 0, 0, 0, -2, 0, 0, 0, 6, 0, 0, 0,
                            -2, 0, 0, 0, -1, 0, 0, 0, 0])
    filters.append(minus45Edges)

    plus45Edges3x3 = shape2D([1, 1, 1, -1, 0, 1, -1, -1, 0])
    # TODO XXX non-symetric ?
    filters.append(plus45Edges3x3)

    minus45Edges3x3 = shape2D([1, 1, 1, 1, 0, -1, 1, -1, -1])
    filters.append(minus45Edges3x3)

    #Frequencies
    lowPass = shape2D([0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25])
    filters.append(lowPass)

    highPass = shape2D([1, -2, 1, -2, 5, -2, 1, -2, 1])
    filters.append(highPass)

    highPassMean = shape2D([-1, -1, -1, -1, 9, -1, -1, -1, -1])
    filters.append(highPassMean)

    #Compass gradient masks
    northCGM = shape2D([1, 1, 1, 1, -2, 1, -1, -1, -1])
    filters.append(northCGM)

    northeastCGM = shape2D([1, 1, 1, -1, -2, 1, -1, -1, 1])
    filters.append(northeastCGM)

    eastCGM = shape2D([-1, 1, 1, -1, -2, 1, -1, 1, 1])
    filters.append(eastCGM)

    southeastCGM = shape2D([-1, -1, 1, -1, -2, 1, 1, 1, 1])
    filters.append(southeastCGM)

    southCGM = shape2D([-1, -1, -1, 1, -2, 1, 1, 1, 1])
    filters.append(southCGM)

    southwestCGM = shape2D([-1, -1, -1, 1, -2, 1, 1, 1, 1])
    filters.append(southwestCGM)

    westCGM = shape2D([-1, 1, -1, 1, -2, -1, 1, 1, -1])
    filters.append(westCGM)

    northwestCGM = shape2D([1, 1, 1, 1, -2, -1, 1, -1, -1])
    filters.append(northwestCGM)

    #log
    logM = shape2D([0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 2, -16, 2, 1, 0, 1, 2, 1,
                    0, 0, 0, 1, 0, 0])
    filters.append(logM)

    #application of log and laplacian
    logLaplacianed = shape2D([0, 0, 1, 1, 1, 0, 0, 0, 1, 4, -4, 4, 1, 0, 1,
                              4, -18, -25, -18, 4, 1, 1, -4, -25, 140, -25,
                              -4, 1, 1, 4, -18, -25, -18, 4, 1, 0, 1, 4, -4,
                              4, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    filters.append(logLaplacianed)

    #Misc.
    #--jahne,  pratt
    misc1 = shape2D([1, -2, 1, -2, 4, -2, 1, -2, 1])
    filters.append(misc1)

    misc2 = shape2D([1, 1, 1, 1, -7, 1, 1, 1, 1])
    filters.append(misc2)

    misc3 = shape2D([0, -1, 0, -1, 5, -1, 0, -1, 0])
    filters.append(misc3)

    #--lines chittineni
    misc4 = shape2D([-1, -1, -1, 2, 2, 2, -1, -1, -1])
    filters.append(misc4)

    misc5 = shape2D([-1, -1, 2, -1, 2, -1, 2, -1, -1])
    filters.append(misc5)

    misc6 = shape2D([-1, 2, -1, -1, 2, -1, -1, 2, -1])
    filters.append(misc6)

    misc7 = shape2D([2, -1, -1, -1, 2, -1, -1, -1, 2])
    filters.append(misc7)

    return filters


def custom_finite_3_same_filter():
    return Finite3SameFilter(custom_filters())


if __name__ == "__main__":
    cuFilt = custom_finite_3_same_filter()
    print len(cuFilt)
