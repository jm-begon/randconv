# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
""" """
import numpy as np

try:
    import Image
except ImportError:
    from PIL import Image

from .numpy_pil_converter import NumpyPILConvertor


__all__ = ["ConvolutionalExtractor"]


class ConvolutionalExtractor:
    """
    ======================
    ConvolutionalExtractor
    ======================
    A :class:`ConvolutionalExtractor` extract features from images. It
    proceeds in 3 steps :

    1. Filtering
        It uses a :class:`FiniteFilter` to generate filters. Thoses filters
        are then applied by a :class:`Convolver` to the given image thus
        creating several new images (one per filter). Let us call them
        *image2*.
    2. Pooling
        Each new *image2* is aggregated by a :class:`Aggregator`, yielding one
        image (let us call them *image3*) by processed *image2*s.
    3. Subwindow extraction
        On each *image3* the same subwindows are extracted giving the set
        of *image4*. This set contains nb_filter*nb_subwindow images

    Note
    ----
    - Compatibily
        The :class:`FiniteFilter` and the :class:`Convolver` must be compatible
        with the kind of image provided !
        Example : the image is a RGB PIL image or RGB numpy array with a
        :class:`RGBConvolver` and a :class:`Finite3Filter`

    - Image representation
        See :mod:`ImageBuffer` for more information

    - It is also possible to include the original image in the process
    """

    def __init__(self, finite_filter, convolver, multi_sw_extractor, multi_pooler,
                 include_original_image=False):
        """
        Construct a :class:`ConvolutionalExtractor`

        Parameters
        ----------
        finite_filter : :class:`FiniteFilter`
            The filter generator and holder
        convolver : :class:`Convolver`
            The convolver which will apply the filter. Must correspond with
            the filter generator and the image type
        pooler : :class:`MultiPooler`
            The :class:`MultiPooler`which will carry the spatial poolings
            **Note** : the spatial poolings must produce ouputs of the same
            shape !
        include_original_image : boolean (default : False)
            Whether or not to include the original image for the subwindow
            extraction part
        """
        self._finite_filter = finite_filter
        self._convolver = convolver
        self._sw_extractor = multi_sw_extractor
        self._multi_pooler = multi_pooler
        self._include_image = include_original_image

    def extract(self, image):
        """
        Extract feature from the given image

        Parameters
        ----------
        image : :class:`PIL.Image` or preferably a numpy array
            The image to process

        Return
        ------
        all_subwindow : a list of lists of subwindows
            The element e[i][j] is a numpy array correspond to the ith
            subwindow of the jth filter.
            If the original image is included, it correspond to the first
            (0th) filter.
        """

        #Converting image in the right format
        convertor = NumpyPILConvertor()
        image = convertor.pil_to_numpy(image)

        filtered = []

        #Including the original image if desired
        if self._include_image:
            pooled_ls = self._multi_pooler.multipool(image)
            for pooled in pooled_ls:
                filtered.append(pooled)
        #Applying the filters & Aggregating
        for filt in self._finite_filter:
            #Filtering
            npTmp = self._convolver(image, filt)
            #Aggregating
            pooled_ls = self._multi_pooler.multipool(npTmp)
            for pooled in pooled_ls:
                filtered.append(pooled)

        #Refreshing the boxes
        shape = filtered[0].shape
        self._sw_extractor.refresh(shape[1], shape[0])  # width, height

        #Extracting the subwindows
        nb_filterss = len(self._finite_filter)
        nbSubWindow = len(self._sw_extractor)
        nbPoolers = len(self._multi_pooler)
        nbImageFactor = nb_filterss*nbPoolers
        if self._include_image:
            nbImageFactor += nbPoolers

        all_subwindows = [[0] * nbImageFactor for i in xrange(nbSubWindow)]

        for col, numpies in enumerate(filtered):
            #converting image to the right format
            img = convertor.numpy_to_pil(numpies)
            #Extracting the subwindows s.s.
            subwindows = self._sw_extractor.extract(img)
            for row in xrange(nbSubWindow):
                all_subwindows[row, col] = convertor.pil_to_numpy(subwindows[row])

        return all_subwindows

    def get_filters(self):
        """
        Return the filters used to process the image

        Return
        ------
        filters : iterable of numpy arrays
            The filters used to process the image, with the exclusion
            of the identity filter if the raw image was included
        """
        return self._finite_filter

    def get_poolers(self):
        """
        Return
        ------
        multi_pooler : class:`MultiPooler`
            The poolers
        """
        return self._multi_pooler

    def is_image_included(self):
        """
        Whether the raw image was included

        Return
        ------
        isIncluded : boolean
            True if the raw image was included
        """
        return self._include_image

    def get_nb_subwindows(self):
        """
        Return the number of subwindows extracted
        """
        return self._sw_extractor.nb_subwidows()

    def get_final_size_per_subwindow(self):
        return self._sw_extractor.get_final_size()


