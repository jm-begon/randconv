# -*- coding: utf-8 -*-
"""
An :class:`Aggregator` maps a 2D or 3D numpy array to a smaller one by
applying a function on a contiguous bloc of pixels.
This module contains several predefined Aggregators.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__date__ = "20 January 2015"


import numpy as np
from math import ceil, floor

from .pooler import Pooler

__all__ = ["Aggregator", "FixSizeAggregator", "AverageAggregator",
           "MaximumAggregator", "MinimumAggregator"]


class Aggregator(Pooler):
    """
    ==========
    Aggregator
    ==========

    An Aggregator maps a 2D or 3D numpy array to a smaller one (of same
    dimentionnality) by applying a function on a contiguous bloc of pixels.

    **2D array** : The Aggregator defines a neighborhood box and a function.
    The function map the content of the box into a single value. The box moves
    around the array so as to include each value only once. Therefore,
    the result is a smaller array.

    **3D array** : Proceeds in the same fashion as 2D by iterating over the 3rd
    dimension. The results are appended so as to form a 3D output.
    """

    def __init__(self, aggregating_function, neighborhood_width,
                 neighborhood_height, include_offset=False):
        """
        Construct an :class:`Aggregator`

        Parameters
        ----------
        aggregating_function : callable
            A function which maps a 2D (numpy) array into a single value
        neighborhood_width : int
            The neighborhood window width
        neighborhood_height : int
            The neighborhood window height
        include_offset: boolean (default : False)
            Whether to include the offset or treat it separately. The offset
            is the remaining part of the division of the array size by
            neigborhood window size.
                if True, the last neighborhood windows are enlarge to treat
                the offset
                if False, the offset is treated separately
        """
        self._init(aggregating_function, neighborhood_width,
                   neighborhood_height, include_offset)

    def _init(self, aggregating_function, neighborhood_width,
              neighborhood_height, include_offset=False):
#        """
#        Construct an :class:`Aggregator`
#
#        Parameters
#        ----------
#        aggregating_function : callable
#            A function which maps a 2D (numpy) array
#        neighborhood_width : int
#            The neighborhood window width
#        neighborhood_height : int
#            The neighborhood window height
#        include_offset: boolean
#            Whether to include the offset or treat it separately. The offset
#            is the remaining part of the division of the array size by
#            neigborhood window size.
#                if True, the last neighborhood windows are enlarge to treat
#                the offset
#                if False, the offset is treated separately
#        """
        self._agg_func = aggregating_function
        self._neighb_width = neighborhood_width
        self._neigh_height = neighborhood_height
        self._include_offset = include_offset

    def _get_boxes(self, np_width, np_height):
#        """
#        Return the number of horizontal and vertical division as well as the
#        boxes use by the Aggregator
#
#        Parameters
#        ----------
#        np_width : int
#            The width of the original array
#        np_height: int
#            The hieght of the original array
#
#        Return
#        ------
#        nb_hz_steps : int
#            the number of horizontal steps
#        nb_v_steps : int
#            the number of vertical steps
#        boxes : list (of box)
#            A box is a tuple (Hz coord, V coord, Hz origin, V origin,
#            Hz destination, V destination)
#            origin isincluded and destination excluded : [origin, destination)
#        """
        nb_hz_steps = np_width/self._neighb_width
        nb_v_steps = np_height/self._neigh_height

        if self._include_offset:
            nb_hz_steps = int(floor(nb_hz_steps))
            nb_v_steps = int(floor(nb_v_steps))
        else:
            nb_hz_steps = int(ceil(nb_hz_steps))
            nb_v_steps = int(ceil(nb_v_steps))

        boxes = []
        #All internal
        for i in range(nb_hz_steps-1):
            for j in range(nb_v_steps-1):
                hzOrigin = i*self._neighb_width
                hzDest = hzOrigin+self._neighb_width
                vOrigin = j*self._neigh_height
                vDest = vOrigin+self._neigh_height
                boxes.append((i, j, hzOrigin, vOrigin, hzDest, vDest))

            #Last column
            hzOrigin = i*self._neighb_width
            hzDest = hzOrigin+self._neighb_width
            vOrigin = (nb_v_steps-1)*self._neigh_height
            vDest = np_height
            boxes.append((i, nb_v_steps-1, hzOrigin, vOrigin, hzDest, vDest))

        #Last line
        for j in range(nb_v_steps-1):
            hzOrigin = (nb_hz_steps-1)*self._neighb_width
            hzDest = np_width
            vOrigin = j*self._neigh_height
            vDest = vOrigin+self._neigh_height
            boxes.append((nb_hz_steps-1, j, hzOrigin, vOrigin, hzDest, vDest))

        #Last line+col
        hzOrigin = (nb_hz_steps-1)*self._neighb_width
        hzDest = np_width
        vOrigin = (nb_v_steps-1)*self._neigh_height
        vDest = np_height
        boxes.append((nb_hz_steps-1, nb_v_steps-1, hzOrigin, vOrigin,
                      hzDest, vDest))

        return nb_hz_steps, nb_v_steps, boxes

    def _do_aggregate(self, array2D):
#        """
#        Aggregate the 2D array
#
#        Parameters
#        ----------
#        array2D : 2D numpy array
#            The array to aggregate
#
#        Return
#        ------
#        aggregated : 2D numpy array
#            The aggregated numpy array
#        """
        np_height, np_width = array2D.shape
        nb_hz_steps, nb_v_steps, boxes = self._get_boxes(np_width, np_height)
        aggregArray = np.zeros((nb_v_steps, nb_hz_steps))

        for box in boxes:
            newCol, new_row, originCol, originRow, destCol, destRow = box
            aggregArray[new_row, newCol] = self._agg_func(
                array2D[originRow:destRow, originCol:destCol])

        return aggregArray

    def aggregate(self, np_array):
        """
        Aggregate the `np_array`

        Parameters
        -----------
        np_array : 2D or 3D numpy array
            The array to aggregate

        Return
        -------
        aggregated : 2D or 3D numpy array (depending on `np_array`)
            The aggregated array
        """
        if len(np_array.shape) < 2 or len(np_array.shape) > 3:
            raise ValueError

        if len(np_array.shape) == 2:
            #2D case
            return self._do_aggregate(np_array)

        #3D case
        layers = []
        for i in range(np_array.shape[2]):
            layers.append(self._do_aggregate(np_array[:, :, i]))

        return np.dstack(layers)

    def __call__(self, np_array):
        """
        Delegate to :meth:`aggregate`
        """
        return self.aggregate(np_array)

    def pool(self, np_array):
        """
        Delegate to :meth:`aggregate`
        """
        return self.aggregate(np_array)


class FixSizeAggregator(Aggregator):
    """
    =================
    FixSizeAggregator
    =================
    A :class:`FixSizeAggregator` operates on arrays of predefined sizes.

    see :class:`Aggregator`
    """

    def __init__(self, aggregating_function, neighborhood_width,
                 neighborhood_height, np_width, np_height,
                 include_offset=False):
        """
        Construct a :class:`FixSizeAggregator`

         Parameters
        -----------
        aggregating_function : callable
            A function which maps a 2D (numpy) array
        neighborhood_width : int
            The neighborhood window width
        neighborhood_height : int
            The neighborhood window height
        np_width : int
            The width of the array this :class:`Aggregator` will treat
        np_height : int
            The height of the array this :class:`Aggregator` will treat
        include_offset: boolean
            Whether to include the offset or treat it separately. The offset
            is the remaining part of the division of the array size by
            neigborhood window size.
                if True, the last neighborhood windows are enlarge to treat
                the offset
                if False, the offset is treated separately
        """
        self._init2(aggregating_function, neighborhood_width,
                    neighborhood_height, include_offset, np_width, np_height)

    def _init2(self, aggregating_function, neighborhood_width,
               neighborhood_height, include_offset, np_width, np_height):
#        """
#        Construct a :class:`FixSizeAggregator`
#
#         Parameters
#        -----------
#        aggregating_function : callable
#            A function which maps a 2D (numpy) array
#        neighborhood_width : int
#            The neighborhood window width
#        neighborhood_height : int
#            The neighborhood window height
#        np_width : int
#            The width of the array this :class:`Aggregator` will treat
#        np_height : int
#            The height of the array this :class:`Aggregator` will treat
#        include_offset: boolean
#            Whether to include the offset or treat it separately. The offset
#            is the remaining part of the division of the array size by
#            neigborhood window size.
#                if True, the last neighborhood windows are enlarge to treat
#                the offset
#                if False, the offset is treated separately
#        """
        self._init(aggregating_function, neighborhood_width,
                   neighborhood_height, include_offset)
        (self._nb_hz_steps, self._nb_v_steps,
         self._boxes) = self._get_boxes(np_width, np_height)

    def _do_aggregate(self, array2D):
#        """
#        Aggregate the 2D array
#
#        Parameters
#        ----------
#        array2D : 2D numpy array
#            The array to aggregate
#
#        Return
#        ------
#        aggregated : 2D numpy array
#            The aggregated numpy array
#        """
        aggregArray = np.zeros((self._nb_v_steps, self._nb_hz_steps))

        for box in self._boxes:
            newCol, new_row, originCol, originRow, destCol, destRow = box
            aggregArray[new_row, newCol] = self._agg_func(
                array2D[originRow:destRow, originCol:destCol])

        return aggregArray


class AverageAggregator(FixSizeAggregator):
    """
    =================
    AverageAggregator
    =================
    A :class:`AverageAggregator` computes the arithmetic average of the
    neighborhood

    See also :class:`Aggregator`, :class:`FixSizeAggregator`
    """

    def __init__(self, neighborhood_width, neighborhood_height,
                 np_width, np_height, include_offset=False):
        """
        Construct a :class:`AverageAggregator`

         Parameters
        -----------
        neighborhood_width : int
            The neighborhood window width
        neighborhood_height : int
            The neighborhood window height
        np_width : int
            The width of the array this :class:`Aggregator` will treat
        np_height : int
            The height of the array this :class:`Aggregator` will treat
        include_offset: boolean
            Whether to include the offset or treat it separately. The offset
            is the remaining part of the division of the array size by
            neigborhood window size.
                if True, the last neighborhood windows are enlarge to treat
                the offset
                if False, the offset is treated separately
        """
        self._init2(self._mean, neighborhood_width, neighborhood_height,
                    include_offset, np_width, np_height)

    def _mean(self, x):
        return x.mean()


class MaximumAggregator(FixSizeAggregator):
    """
    =================
    MaximumAggregator
    =================
    A :class:`MaximumAggregator` computes the maximum of the neighborhood

    See also :class:`Aggregator`, :class:`FixSizeAggregator`
    """

    def __init__(self, neighborhood_width, neighborhood_height, np_width,
                 np_height, include_offset=False):
        """
        Construct a :class:`MaximumAggregator`

         Parameters
        -----------
        neighborhood_width : int
            The neighborhood window width
        neighborhood_height : int
            The neighborhood window height
        np_width : int
            The width of the array this :class:`Aggregator` will treat
        np_height : int
            The height of the array this :class:`Aggregator` will treat
        include_offset: boolean
            Whether to include the offset or treat it separately. The offset
            is the remaining part of the division of the array size by
            neigborhood window size.
                if True, the last neighborhood windows are enlarge to treat
                the offset
                if False, the offset is treated separately
        """
        self._init2(self._max, neighborhood_width, neighborhood_height,
                    include_offset, np_width, np_height)

    def _max(self, x):
        return x.max()


class MinimumAggregator(FixSizeAggregator):
    """
    =================
    MinimumAggregator
    =================
    A :class:`MinimumAggregator` computes the minimum of the neighborhood

    See also :class:`Aggregator`, :class:`FixSizeAggregator`
    """

    def __init__(self, neighborhood_width, neighborhood_height, np_width,
                 np_height, include_offset=False):
        """
        Construct a :class:`MinimumAggregator`

         Parameters
        -----------
        neighborhood_width : int
            The neighborhood window width
        neighborhood_height : int
            The neighborhood window height
        np_width : int
            The width of the array this :class:`Aggregator` will treat
        np_height : int
            The height of the array this :class:`Aggregator` will treat
        include_offset: boolean
            Whether to include the offset or treat it separately. The offset
            is the remaining part of the division of the array size by
            neigborhood window size.
                if True, the last neighborhood windows are enlarge to treat
                the offset
                if False, the offset is treated separately
        """
        self._init2(self._min, neighborhood_width, neighborhood_height,
                    include_offset, np_width, np_height)

    def _min(self, x):
        return x.min()
