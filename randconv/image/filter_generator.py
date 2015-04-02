# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 22 2014
"""
A set of filter generator and holder.

Filter
------
A filter is 2D numpy array which can be used to filter an image with a
convolution operator
"""
import numpy as np
import itertools
from sklearn.utils import check_random_state
from ..util import IntegerUniformGenerator




class FilterGenerator:
    """
    ===============
    FilterGenerator
    ===============
    A base class which generate randomly filters with specified features.
    A filter is a 2D numpy array.

    Constants
    ---------
    Normalisation parameters :
    NORMALISATION_NONE : indicate that no normalisation is required.
    NORMALISATION_MEAN : indicate that filter should be normalised to have
    a mean value of 0.
    NORMALISATION_VAR : indicate that filter should be normalised to have
    variance of 1.
    NORMALISATION_MEANVAR : indicate that filter should be normalised to have
    a mean of 0 and a variance of 1.

    Constructor parameters
    ----------------------
    num_val_generator : :class:`NumberGenerator`
        The generator which will draw the values of the filter
    num_size_generator : :class:`NumberGenerator`
        The generator which will draw the size of the filter.
        Must generate positive integers.
    square_sized : boolean (default : True)
        Whether or not the filter should have the same width as height
    normalisation : int (normalisation parameter : {NORMALISATION_NONE,
    NORMALISATION_MEAN, NORMALISATION_VAR, NORMALISATION_MEANVAR}) (
    default : NORMALISATION_NONE)
        The normalisation to apply to the filter
    """
    NORMALISATION_NONE = 0
    NORMALISATION_MEAN = 1
    NORMALISATION_VAR = 2
    NORMALISATION_MEANVAR = 3
    NORMALISATION_SUMTO1 = 4
    EPSILON = 10**-15

    def __init__(self, num_val_generator, num_size_generator,
                 square_sized=True, normalisation=NORMALISATION_NONE):

        self._value_generator = num_val_generator
        self._normalisation = normalisation
        self._size_generator = num_size_generator
        self._sqr = square_sized

    def _get_size(self):
        """
        Generate the height and width
        """
        height = self._size_generator.get_number()
        if self._sqr:
            width = height
        else:
            width = self._size_generator.get_number()
        return height, width

    def _normalize(self, filt):
        """
        Normalize the filter according to the instance policy

        Parameters
        ----------
        filt : 2D numpy array
            The filter to normalize

        Return
        ------
        normalizedFilter : 2D numpy array of the same shape
            the normalized filter
        """
        if self._normalisation is None:
            return filt

        if self._normalisation == FilterGenerator.NORMALISATION_NONE:
            return filt

        #Normalisation
        if self._normalisation == FilterGenerator.NORMALISATION_SUMTO1:
            return filt/sum(filt)
        if (self._normalisation == FilterGenerator.NORMALISATION_MEAN or
                self._normalisation == FilterGenerator.NORMALISATION_MEANVAR):
            filt = filt - filt.mean()

        if (self._normalisation == FilterGenerator.NORMALISATION_VAR or
                self._normalisation == FilterGenerator.NORMALISATION_MEANVAR):
            stdev = filt.std()
            if abs(stdev) > FilterGenerator.EPSILON:
                filt = filt/stdev
        return filt

    def __iter__(self):
        return self

    def next(self):
        """
        Return a newly generated filter
        """
        #Generate the size
        height, width = self._get_size()
        #Fill in the values
        linearFilter = self.create_filter(height, width)
        return self._normalize(linearFilter)

    def create_filter(self, height, width):
        """
        Create the a filter of the according dimension

        Parameters
        ----------
        height : int > 0
            The height of the filter
        width : int > 0
            The width of the filter

        Return
        ------
        filter : 2D numpy array
            The generated filter
        """
        linearFilter = np.zeros((height, width))
        for i in xrange(height):
            for j in xrange(width):
                linearFilter[i][j] = self._value_generator.get_number()
        return linearFilter


class FixSizeFilterGenerator(FilterGenerator):
    """
    ======================
    FixSizeFilterGenerator
    ======================
    Generate filters of constant size
    """
    def __init__(self, num_val_generator, height, width,
                 normalisation=FilterGenerator.NORMALISATION_NONE):
        """
        Construct a :class:`FixSizeFilterGenerator`

        Parameters
        ----------
        num_val_generator : :class:`NumberGenerator`
            The generator which will draw the values of the filter
        height : int > 0
            The height (number of rows) of the filter
        width : int > 0
            The width (number of columns) of the filter
        square_sized : boolean (default : True)
            Whether or not the filter should have the same width as height
        normalisation : int (normalisation parameter : {NORMALISATION_NONE,
        NORMALISATION_MEAN, NORMALISATION_VAR, NORMALISATION_MEANVAR}) (
        default : NORMALISATION_NONE)
            The normalisation to apply to the filter
        """
        self._value_generator = num_val_generator
        self._height = height
        self._width = width
        self._normalisation = normalisation

    def _get_size(self):
        return self._height, self._width


class IdPerturbatedFG(FilterGenerator):
    """
    ===============
    IdPerturbatedFG
    ===============
    An class:`IdPerturbatedFG` instance produces
    filter by randomly pertubating the identity filter of a given size

    Constructor parameters
    ----------------------
    num_val_generator : :class:`NumberGenerator`
        The generator which will draw the values of the filter
    num_size_generator : :class:`NumberGenerator`
        The generator which will draw the size of the filter.
        Must generate positive odd integers.
    square_sized : boolean (default : True)
        Whether or not the filter should have the same width as height
    normalisation : int (normalisation parameter : {NORMALISATION_NONE,
    NORMALISATION_MEAN, NORMALISATION_VAR, NORMALISATION_MEANVAR}) (
    default : NORMALISATION_NONE)
        The normalisation to apply to the filter
    """

    def __init__(self, num_val_generator, num_size_generator,
                 square_sized=True,
                 normalisation=FilterGenerator.NORMALISATION_NONE):

        FilterGenerator.__init__(self, num_val_generator,
                                 num_size_generator,
                                 square_sized,
                                 normalisation)

    def create_filter(self, height, width):
        """Overload"""
        #Create id filter
        linearFilter = np.zeros((height, width))
        hCenter = height//2
        wCenter = width//2
        linearFilter[hCenter, wCenter] = 1
        #Perturbate
        for i in xrange(height):
            for j in xrange(width):
                linearFilter[i][j] += self._value_generator.get_number()
        return linearFilter


class IdMaxL1DistPerturbFG(IdPerturbatedFG):

    def __init__(self, num_val_generator, num_size_generator,
                 max_dist, square_sized=True,
                 normalisation=FilterGenerator.NORMALISATION_NONE,
                 shuffling_seed=None):

        IdPerturbatedFG.__init__(self, num_val_generator,
                                 num_size_generator, square_sized,
                                 normalisation)
        self._shuffler = check_random_state(shuffling_seed)
        self._max_dist = max_dist

    def create_filter(self, height, width):
        #Create id filter
        linearFilter = np.zeros((height, width))
        hCenter = height//2
        wCenter = width//2
        linearFilter[hCenter, wCenter] = 1

        max_dist = self._max_dist
        #Shuffling coordinates
        ls = [x for x in itertools.product(xrange(height), xrange(width))]
        self._shuffler.shuffle(ls)
        #Pertubating
        for h, w in ls:
            if max_dist < 0:
                break
            val = self._value_generator.get_number(-max_dist, max_dist)
            max_dist -= abs(val)
            linearFilter[h][w] += val
        return linearFilter


class StratifiedFG(FilterGenerator):
    """
    Constructor parameters
    ----------------------
    min_val : number
        The minimum value of a component (included)
    max_val : number
        The maximum value of a component (excluded)
    nb_cells : int > 0
        The number of cells/The number of division on the min_val-max_val
        segment
    perturbation_generator : class:`NumberGenerator` (must be [0, 1) range)
        The generator which will produce the perturbation
    cell_seed : int (default : None = random)
        The seed for the random generator which will draw the chosen
        cells
    """

    def __init__(self, min_val, max_val, nb_cells, perturbation_generator,
                 num_size_generator, square_sized=True,
                 normalisation=FilterGenerator.NORMALISATION_NONE,
                 cell_seed=None):

        FilterGenerator.__init__(self, perturbation_generator,
                                 num_size_generator, square_sized,
                                 normalisation)
        self._min = min_val
        self._max = max_val
        self._nb_cells = nb_cells
        self._cellChooser = IntegerUniformGenerator(0, nb_cells, cell_seed)

    def create_filter(self, height, width):
        """Overload"""
        linearFilter = np.zeros((height, width))
        for i in xrange(height):
            for j in xrange(width):
                #Locating the middle of the normalized cell
                inc = 1./self._nb_cells
                cell = self._cellChooser.get_number()
                start = inc*cell
                end = start+inc
                middle = (start+end)/2.
                #Applying the perturbation
                perturbation = self._value_generator.get_number(0, inc)
                val = middle+perturbation
                #Scaling & shifting
                valRange = self._max - self._min
                val = self._min + valRange*val
                #Assigning the value
                linearFilter[i][j] = val
        return linearFilter


class SparsityDecoratorFG(FilterGenerator):

    def __init__(self, filter_generator, sparse_proba, seed=None):
        self._fg = filter_generator
        self._numGen = NumberGenerator(0, 1, seed)
        self._prob = sparse_proba

    def next(self):
        filt = self._fg.next()
        height, width = filt.shape
        for h in xrange(height):
            for w in xrange(width):
                if self._numGen.get_number(0, 1) < self._prob:
                    filt[h][w] = 0
        return filt


class FiniteFilter:
    """
    ============
    FiniteFilter
    ============

    A :class:`FiniteFilter` is a container of filters.

    Constructor parameters
    ----------------------
    filter_generator : :class:`FilterGenerator` or any container holding
    filters and disposing of :meth:`next` method which produces a sequence
    of filters or an iterable of filters
        The generator which will produce the filters
    nb_filters : int > 0 or None (default : None)
        The number of filters to generate. If None, all the filters of
        `filter_generator` are drawn.
    """

    def __init__(self, filter_generator, nb_filters=None):
        filters = []
        if nb_filters is not None:
            if hasattr(filter_generator, "__getitem__"):
                for i in xrange(nb_filters):
                    filters.append(filter_generator[i])
            else:
                for i in xrange(nb_filters):
                    filters.append(filter_generator.next())
        else:
            #Then we must iterate
            for filt in filter_generator:
                filters.append(filt)

        self._filters = filters

    def __iter__(self):
        return iter(self._filters)

    def __len__(self):
        return len(self._filters)


class Finite3Filter(FiniteFilter):
    """
    =============
    Finite3Filter
    =============

    A :class:`Finite3Filter` is a container of filter triplets.

    Constructor parameters
    ----------------------
    filter_generator : :class:`FilterGenerator` or any container holding
    filters and disposing of :meth:`next` method which produces a sequence
    of filters or an iterable of filters
        The generator which will produce the filters
    nb_filters : int > 0 or None (default : None)
        The number of filters to generate. If None, all the filters of
        `filter_generator` are drawn.
    """
    def __init__(self, filter_generator, nb_filters=None):
        filters = []
        if nb_filters is not None:
            if hasattr(filter_generator, "__getitem__"):
                for i in xrange(0, nb_filters, 3):
                    f1 = filter_generator[i]
                    f2 = filter_generator[i+1]
                    f3 = filter_generator[i+2]
                    filters.append((f1, f2, f3))
            else:
                for i in xrange(nb_filters):
                    filters.append(filter_generator.next(),
                                   filter_generator.next(),
                                   filter_generator.next())
        else:
            #Then we must iterate
            for filt in filter_generator:
                filters.append(filt)
            filters2 = []
            for i in xrange(0, len(filters), 3):
                    f1 = filters[i]
                    f2 = filters[i+1]
                    f3 = filters[i+2]
                    filters2.append((f1, f2, f3))
            filters = filters2

        self._filters = filters


class Finite3SameFilter(Finite3Filter):
    """
    =================
    Finite3SameFilter
    =================

    A :class:`Finite3SameFilter` is a container of filter triplets where
    each filter of a triplet are the same

    Constructor parameters
    ----------------------
    filter_generator : :class:`FilterGenerator` or any container holding
    filters and disposing of :meth:`next` method which produces a sequence
    of filters or an iterable of filters
        The generator which will produce the filters
    nb_filters : int > 0 or None (default : None)
        The number of filters to generate. If None, all the filters of
        `filter_generator` are drawn.
    """

    def __init__(self, filter_generator, nb_filters=None):
        filters = []
        if nb_filters is not None:
            if hasattr(filter_generator, "__getitem__"):
                for i in xrange(nb_filters):
                    filt = filter_generator[i]
                    filters.append((filt, filt, filt))
            else:
                for i in xrange(nb_filters):
                    filt = filter_generator.next()
                    filters.append((filt, filt, filt))
        else:
            #Then we must iterate
            for filt in filter_generator:
                filters.append((filt, filt, filt))

        self._filters = filters


def OrderedMFF(filter_generators, nb_filterss):
    count = 0
    ls = []
    for fg in filter_generators:
        for filt in fg:
            if count == nb_filterss-1:
                return ls
            count += 1
            ls.append(filt)
    return ls


#def RoundRobinMFF(filter_generators, nb_filterss, probVect=None):
#    if probVect is None:
#        ls = [None]*nb_filterss
#        for i in xrange(nb_filterss):
#            ls[i] = filter_generators[i % len(filter_generators)].next()
#        return ls
#    else:
#        pass  # TODO XXX



