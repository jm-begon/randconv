# -*- coding: utf-8 -*-
"""
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__date__ = "20 January 2015"


import numpy as np
cimport numpy as np
cimport cython

from .pooler import Pooler


class FastMWPooler(Pooler):
    """
    Construc a class:`Pooler` instance

    Parameters
    ----------
    height : int > 0 odd number
        the window height
    width : int > 0 odd number
        the window width
    """
    def __init__(self, fastFunction, height, width):

        self._function = fastFunction
        self._windowHalfHeight = height//2
        self._windowHalfWidth = width//2

    def pool(self, np_array):
        if np_array.ndim == 2:
            return self._function(np_array, self._windowHalfHeight, self._windowHalfWidth)
        ls = []
        cdef unsigned int i
        for i in xrange(np_array.shape[2]):
            ls.append(self._function(np_array[:,:,i], self._windowHalfHeight, self._windowHalfWidth))
        return np.dstack(ls)


class FastMWAvgPooler(FastMWPooler):

    def __init__(self, height, width):
        """
        Construc a class:`Pooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        FastMWPooler.__init__(avg_pooling, height, width)

class FastMWMaxPooler(FastMWPooler):

    def __init__(self, height, width):
        """
        Construc a class:`Pooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        FastMWPooler.__init__(max_pooling, height, width)


class FastMWMinPooler(FastMWPooler):

    def __init__(self, height, width):
        """
        Construc a class:`Pooler` instance

        Parameters
        ----------
        height : int > 0 odd number
            the window height
        width : int > 0 odd number
            the window width
        """
        FastMWPooler.__init__(min_pooling, height, width)




@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def avg_pooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol, counter
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double acc
    height = img.shape[0]
    width = img.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((height, width))

    for row in xrange(height):
        for col in xrange(width):

            rMin = row - windowHalfHeight
            rMax = row + windowHalfHeight
            cMin = col - windowHalfWidth
            cMax = col + windowHalfWidth

            u = rMin
            d = rMax + 1  # Inclusive
            l = cMin
            r = cMax + 1  # Inclusive
            if u < 0:
                u = 0
            if l < 0:
                l = 0
            if d > height:
                d = height
            if r > width:
                r = width

            counter = 0
            acc = 0.

            for subrow in xrange(u, d):
                for subcol in xrange(l, r):
                    acc = acc + img[subrow, subcol]
                    counter = counter + 1

            result[row, col] = acc/counter

    return result

@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def max_pooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double max_val
    height = img.shape[0]
    width = img.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((height, width))

    for row in xrange(height):
        for col in xrange(width):

            rMin = row - windowHalfHeight
            rMax = row + windowHalfHeight
            cMin = col - windowHalfWidth
            cMax = col + windowHalfWidth

            u = rMin
            d = rMax + 1  # Inclusive
            l = cMin
            r = cMax + 1  # Inclusive
            if u < 0:
                u = 0
            if l < 0:
                l = 0
            if d > height:
                d = height
            if r > width:
                r = width

            max_val = img[u, l]

            for subrow in xrange(u, d):
                for subcol in xrange(l, r):
                    if img[subrow, subcol] > max_val:
                        max_val = img[subrow, subcol]

            result[row, col] = max_val

    return result


@cython.wraparound(False)  # Turn off wrapping capabilities (speed up)
@cython.boundscheck(False)  # Turn off bounds-checking for entire function (speed up)
def min_pooling(np.ndarray[np.float64_t, ndim=2] img,
               int windowHalfHeight,
               int windowHalfWidth):

    cdef unsigned int height, width, subrow, subcol
    cdef int   row, col, u, d, l, r, rMin, rMax, cMin, cMax
    cdef double min_val
    height = img.shape[0]
    width = img.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((height, width))

    for row in xrange(height):
        for col in xrange(width):

            rMin = row - windowHalfHeight
            rMax = row + windowHalfHeight
            cMin = col - windowHalfWidth
            cMax = col + windowHalfWidth

            u = rMin
            d = rMax + 1  # Inclusive
            l = cMin
            r = cMax + 1  # Inclusive
            if u < 0:
                u = 0
            if l < 0:
                l = 0
            if d > height:
                d = height
            if r > width:
                r = width

            min_val = img[u, l]

            for subrow in xrange(u, d):
                for subcol in xrange(l, r):
                    if img[subrow, subcol] < min_val:
                        min_val = img[subrow, subcol]

            result[row, col] = min_val

    return result
