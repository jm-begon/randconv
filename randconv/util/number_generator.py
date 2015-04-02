# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 22 2014
"""
A set of number generator
"""
from sklearn.utils import check_random_state
from scipy.stats import norm

__all__ = ["NumberGenerator", "IntegerUniformGenerator", "OddUniformGenerator",
           "GaussianNumberGenerator", "ClippedGaussianRNG",
           "CustomDiscreteNumberGenerator", "ConstantGenerator"]


class NumberGenerator:
    """
    ===============
    NumberGenerator
    ===============
    A random number generator which draws number between two bounds :
    [min, max)
    This base class returns uniform real number

    Constructor parameters
    ----------------------
    min_val : float (default : 0)
        The minimum value from which to draw
    max_val : float (default : 1)
        The maximum value from which to draw
    seed : int or None (default : None)
        if seed is int : initiate the random generator with this seed
    """
    def __init__(self, min_val=0, max_val=1, seed=None):
        self._rand_gen = check_random_state(seed)
        self._min = min_val
        self._max = max_val

    def _do_get_number(self, min_val, max_val):
        """
        Return a random number comprise between [min_val, max_val)

        Parameters
        ----------
        min_val : float
           The minimum value from which to draw
        max_val : float
           The maximum value from which to draw

        Return
        ------
        rand : float
           A random number comprise between [min_val, max_val)
        """
        return min_val+self._rand_gen.rand()*(max_val-min_val)

    def get_number(self, min_val=None,  max_val=None):
        """
        Return a random number comprise between [min_val, max_val)

        Parameters
        ----------
        min_val : float/None (default : None)
            The minimum value from which to draw
            if None : use the instance minimum
        max_val : float/None (default : None)
            The maximum value from which to draw
            if None : use the instance maximum

        Return
        ------
        rand : float
            A random number comprise between [min_val, max_val)
        """
        if min_val is None:
            min_val = self._min
        if max_val is None:
            max_val = self._max
        return self._do_get_number(min_val, max_val)

    def __iter__(self):
        while True:
            yield self.get_number()



class GaussianNumberGenerator(NumberGenerator):

    """
    =======================
    GaussianNumberGenerator
    =======================
    Generate real number from a Gaussian law

    Bound policy
    ------------
    The mean is (min_val + max_val)/2.
    The stdev is computed thanks to the outside_proba

    Constructor parameters
    ----------------------
    min_val : float (default : 0)
        The minimum value from which to draw
    max_val : float (default : 1)
        The maximum value from which to draw
    seed : int or None (default : None)
        if seed is int : initiate the random generator with this seed
    outside_proba : float [0, 1)
        The probability of generating a value which is outside of the range
        [min_val, max_val)
    """

    def __init__(self, min_val=0, max_val=1, seed=None, outside_proba=0.05):
        NumberGenerator.__init__(self, min_val, max_val, seed)
        inRange = 1-(outside_proba/2.)
        self._k = norm.ppf(inRange)

    def _do_get_number(self, min_val, max_val):
        mean = (max_val + min_val)/2.
        std = (mean - min_val)/self._k
        val = self._rand_gen.normal(mean, std)
        return val


class ClippedGaussianRNG(GaussianNumberGenerator):
    """
    =======================
    GaussianNumberGenerator
    =======================
    Generate real number from a Gaussian law, clipped at the given range

    Bound policy
    ------------
    If the value is outside the min-max range, it clipped to the bound
    """
    def __init__(self, min_val=0, max_val=1, seed=None, outside_proba=0.05):
        GaussianNumberGenerator.__init__(self, min_val, max_val, seed,
                                         outside_proba)

    def _do_get_number(self, min_val, max_val):
        val = GaussianNumberGenerator._do_get_number(self, min_val, max_val)
        if val < min_val:
            return min_val
        if val > max_val:
            return max_val
        return val


class CustomDiscreteNumberGenerator(NumberGenerator):
    """
    =============================
    CustomDiscreteNumberGenerator
    =============================
    Generate a number of a predifine set whose elements are given a predifine
    probability.

    Note
    ----
    If the "probabilities" associated to the elements does not sum up, there
    are scaled to do so.

    Constructor parameters
    ----------------------
    lsOfPairs : iterable of pairs (number, probability)
        number : number
            an element of the set from which to draw
        probability : float
            the probability of the element being chosen at each draw
    seed : int or None (default : None)
        if seed is int : initiate the random generator with this seed
    """

    def __init__(self, lsOfPairs, seed=None):

        self._vals = [v for v, p in lsOfPairs]
        probs = [p for v, p in lsOfPairs]
        sumP = sum(probs)
        probs = [p/sumP for p in probs]
        self._cumulProbs = [0]*len(probs)
        for i in xrange(len(probs)-1):
            self._cumulProbs[i+1] = self._cumulProbs[i] + probs[i]
        NumberGenerator.__init__(self, 0, 1, seed)

    def get_number(self, min_val=0, max_val=1):
        prob = self._rand_gen.rand()
        index = self._search(prob)
        return self._vals[index]

    def _search(self, prob):
        """
        Return the index of self._cumulProbs such that
        p \in [self._cumulProbs[index], self._cumulProbs[index+1])
        """
        #Linear search for now
        for i in xrange(len(self._cumulProbs)-1):
            if prob >= self._cumulProbs[i] and prob < self._cumulProbs[i+1]:
                return i
        #Should not happen
        if prob < self._cumulProbs[0]:
            return 0
        return len(self._cumulProbs)-1


class IntegerUniformGenerator(NumberGenerator):
    """
    =======================
    IntegerUniformGenerator
    =======================
    A random number generator which draws number between two bounds :
    [min, max)
    This class returns uniform integer number
    """
    def _do_get_number(self, min_val, max_val):
        return self._rand_gen.randint(min_val, max_val)


class OddUniformGenerator(IntegerUniformGenerator):
    """
    =======================
    IntegerUniformGenerator
    =======================
    A random number generator which draws number between two bounds :
    [min, max)
    This class returns uniform odd integers
    """
    def _do_get_number(self, min_val, max_val):
        if max_val % 2 == 0:
            max_val -= 1
        if min_val % 2 == 0:
            min_val += 1
        return min_val + 2*int(self._rand_gen.rand()*((max_val - min_val)/2+1))


class ConstantGenerator(NumberGenerator):
    """
    =================
    ConstantGenerator
    =================
    A not so random number generator. "Generate" a constant number while
    preserving the :class:`NumberGenerator` interface.
    """
    def __init__(self, constant):
        self._const = constant

    def get_number(self, min_val=None,  max_val=None):
        return self._const
