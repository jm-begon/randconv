# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
A :class:`Convolver` applies a filter to an numpy array. Different shapes
of numpy arrays are tackled by different convolvers
"""

from scipy import signal as sg
import numpy as np



class Convolver:
    """
    =========
    Convolver
    =========
    The base :class:`Convolver` performs classical convolution between
    a 2D numpy array and a 2D filter.

    Note
    ----
    The filter must be 2D numpy array.

    Constructor parameters
    ----------------------
    mode : str {"same", "valid", "full"} (default : "same")
        A string indicating the size of the output:
        - full
            The output is the full discrete linear convolution of the
            inputs.
        - valid
            The output consists only of those elements that do not rely
            on the zero-padding.
        - same
            The output is the same size as in1, centered with respect to
            the ‘full’ output.
    boundary : str {"fill", "wrap", "symm"} (default : "fill")
        A flag indicating how to handle boundaries:
        - fill
            pad input arrays with fillvalue.
        - wrap
            circular boundary conditions.
        - symm
            symmetrical boundary conditions.
    fillvalue : scalar (default : 0)
        Value to fill pad input arrays with.
    """

    def __init__(self, mode="same", boundary="fill", fillvalue=0):

        self._mode = mode
        self._boundary = boundary
        self._fillvalue = fillvalue

    def convolve(self, np_image, npFilter):
        """
        Return the 2D convolution of the image by the filter

        Parameters
        ----------
        np_image : 2D array like structure (usually numpy array)
            The image to filter
        npFilter : 2D array like structure (usually numpy array)
            The filter to apply by convolution

        Return
        ------
        filtered : 2D array
            The result of the convolution
        """
        return sg.convolve2d(np_image, npFilter, self._mode, self._boundary,
                             self._fillvalue)

    def __call__(self, np_image, npFilter):
        """
        Delegates to :meth:`convolve`
        """
        return self.convolve(np_image, npFilter)


class RGBConvolver(Convolver):
    """
    ============
    RGBConvolver
    ============
    The :class:`RGBConvolver` treats each colorband separately by performing
    classical convolution between each colorband and its respective filter.
    A colorband is suppose to be a 2D numpy array as are also the 2D filters.
    """

    def convolve(self, np_image, filters):
        """
        Return the 2D convolution of each colorband by its respective filter.

        Parameters
        ----------
        np_image : 3D numpy array where the dimensions represent respectively
        the height, the width and the colorbands. There must be 3 colorbands.
            The image to filter
        filters : a triplet of filters of the same size. The filters are
        2D array like structure (usually numpy array).
            The respective filters. They are applied in the same order to the
            colorbands.

        Return
        ------
        filtered :3D numpy array where the dimensions represent respectively
        the height, the width and the colorbands
            The result of the convolution of each colorband by its respective
            filter
        """
        red_filter, green_filter, blue_filter = filters
        red, green, blue = np_image[:,:,0], np_image[:,:,1], np_image[:,:,2]

        new_red = sg.convolve2d(red, red_filter, "same")
        new_green = sg.convolve2d(green, green_filter, "same")
        new_blue = sg.convolve2d(blue, blue_filter, "same")

        return np.dstack((new_red, new_green, new_blue))



