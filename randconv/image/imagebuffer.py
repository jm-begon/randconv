# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Feb 23 2014
"""
A set of container for (image, label) pairs with load-on-request capacities
"""
from copy import copy

try:
    import Image
except ImportError:
    from PIL import Image
import numpy as np

from .numpy_pil_converter import NumpyToPILConvertor, PILToNumpyConvertor


__all__ = ["ImageBuffer", "FileImageBuffer", "ImageLoader",
           "NotCMapImageLoader", "NumpyImageLoader"]


def _pilNbBands(img):
    print img
    return len(img.getbands())


def _npNbBands(img):
    _nbColors = 1
    if img.shape != 2:
        _nbColors = img.shape[2]
    return _nbColors


#==============Generic Buffer (from a sequence of images)===============
class ImageBuffer:
    """
    ===========
    ImageBuffer
    ===========
    Base class for :class:`ImageBuffer`. An image buffer also convert the
    images to a given format.

    Simply encapsulates an iterator of (image, label) pairs.

    Format
    ------
    Images can be represented as PIL.Image or as numpy arrays. For a RGB
    numpy array, the first dimension represents the height, the second
    the width and the third the color.

    Constants
    ---------
    PIL_FORMAT : the output format is PIL (as describe above)
    NUMPY_FORMAT : the output format is numpy (as describe above)

    Inheritance
    -----------
    A subclass must call the :meth:`_initImageBuffer` protected method. This
    method initializes the `self._imageSeq` protected instance variable,
    which is a container of (**imageable**, label) pairs.
    - Imageable
        An imageable is whatever object that can be converted into an image
        by the :class:`ImageBuffer` instance. The convertion is the
        responsability of :meth:`_extract`. The subclass maintains coherency
        by overloading this method.
    - Label
        A label is a integer (>=0) representing the class of a image. Image
        with the same class have the same labels.
        The pairs must obviously correspond (the label is the class of the
        imageable which it is paired with)

    Summary :
    1. Call :meth:`_initImageBuffer`
    2. Define what is an imageable for that :class:`ImageBuffer`
    3. Overload :meth:`_extract` to tackle the imageable appropriately

    This base class **imageable**
    -----------------------------
    This base class works directly with image. Thus the **imageable** of this
    class are images and the :meth:`_extract` simply return directly the image
    """
    PIL_FORMAT = 1
    NUMPY_FORMAT = 2

    def __init__(self, imageIterator, outputFormat=PIL_FORMAT):
        """
        Constructs a :class:`ImageBuffer`

        Parameters
        ----------
        imageIterator : iterable of (image, int) pairs
            The container of images with its corresponding label
        outputFormat : int {PIL_FORMAT, NUMPY_FORMAT} (default : PIL_FORMAT)
            The output format to which the image will be converted
        """
        self._initImageBuffer(imageIterator, outputFormat)

    def _initImageBuffer(self, imageIterator, outputFormat):
        self._imageSeq = imageIterator
        self._outputFormat = outputFormat
        if outputFormat == ImageBuffer.PIL_FORMAT:
            self._convertor = NumpyToPILConvertor()
        else:
            self._convertor = PILToNumpyConvertor()

    def size(self):
        """
        Return the size (i.e. the number of (imageable, label) pairs)
        """
        return len(self._imageSeq)

    def get(self, index):
        """
        Return the (image, label) pairs of a given index

        Parameters
        ----------
        index : int >= 0
            The index of the pair to return
        Return
        ------
        pair = (image, label)
        image : PIL.Image or numpy array (depending on the output format)
            The image of the given index
        label : int >=0
            The class label of the image
        """
        imageable, label = self._imageSeq[index]
        img = self._extract(imageable)
        return self._convert(img), label

    def _convert(self, img):
#        """
#        Convert the image to the correct output format
#
#        Parameters
#        ----------
#        img : PIL.Image or numpy array
#            The image to convert
#
#        Return
#        ------
#        converted : PIL.Image or numpy array (depending on the output format)
#            The same image at the right format
#        """
        return self._convertor(img)

    def _extract(self, imageable):
        """
        Transform the imageable in the corresponding image (the output format
        convertion is taken care of by the :meth:`get` method thanks to the
        :meth:`_convert` method)

        This method must be overloaded. It is not called directly however.

        For this base class, imageable are already image and therefore the
        method simply return it argument.

        Parameters
        ----------
        imageable : an **imageable** appropriate for the class
            the imageable to convert into image

        Return
        ------
        image : PIL.Image or numpy array
            The corresponding image
        """
        return imageable

    def __len__(self):
        return self.size()

    def __iter__(self):
        return BufferIterator(self)

    def __getitem__(self, index):
        #If the index is a slice, we return a clone of this object with
        # the sliced pair containers
        if isinstance(index, slice):
            clone = copy(self)
            clone._imageSeq = self._imageSeq[index]
            return clone
        #If it is a real index (int), we return the corresponding object
        else:
            return self.get(index)

    def get_labels(self):
        """
        Return the label vector.
        """
        return [label for _, label in self._imageSeq]

    def nb_bands(self):
        if self._outputFormat == ImageBuffer.PIL_FORMAT:
            return _pilNbBands(self.get(0)[0])
        else:
            return _npNbBands(self.get(0)[0])


#==============Buffer Iterator===============
class BufferIterator:
    """
    ==============
    BufferIterator
    ==============
    An iterator for :class:`ImageBuffer`
    """
    def __init__(self, ImageBuffer):
        self._buffer = ImageBuffer
        self._index = 0

    def __iter__(self):
        return self

    def next(self):
        if self._index >= len(self._buffer):
            raise StopIteration
        else:
            index = self._index
            self._index += 1
            return self._buffer.get(index)


#=================Buffer from file =========================
#--------- Image Loader ------
class ImageLoader:
    """
    ===========
    ImageLoader
    ===========
    A class which can load image files into :class:`PIL.Image`.
    See the PIL library for more information.
    """
    def load(self, imageFile):
        """
        Load a image from a file

        Parameters
        ----------
        imageFile : str or file
            The path to the file

        Return
        ------
        image : PIL.Image
            The corresponding image
        """
        return Image.open(imageFile)

    def __call__(self, imageFile):
        """Delegates to :meth:`load` method"""
        return self.load(imageFile)


class NotCMapImageLoader(ImageLoader):
    """
    ==================
    NotCMapImageLoader
    ==================
    Load image file and convert palette into RGB if necessary
    """
    def load(self, imageFile):
        image = Image.open(imageFile)

        if image.mode == "P":
            image = image.convert("RGB")
        return image


class NumpyImageLoader(ImageLoader):
    """
    ==================
    NumpyImageLoader
    ==================
    Load a numpy file representing an image
    """
    def load(self, numpyFile):
        """
        Load a image from a file

        Parameters
        ----------
        imageFile : str or file
            The path to the file

        Return
        ------
        image : numpy array representing an image
            The corresponding image
        """
        return np.load(numpyFile)


#--------- Buffer -----
class FileImageBuffer(ImageBuffer):
    """
    ===============
    FileImageBuffer
    ===============
    An :class:`ImageBuffer` whose imageable are files. The files are only
    loaded as necessary.
    """

    def __init__(self, fileImageCollection, imageLoader,
                 outputFormat=ImageBuffer.PIL_FORMAT):
        """
        Construct a :class:`FileImageBuffer`.

        Parameters
        ----------
        fileImageCollection : iterable of (str/file, int) pairs
            The container of str/file : path to image file with the
            corresponding labels
        imageLoader : :class:`ImageLoader`
            The loader to use to load the image (it must be appropriate to
            the image files provided)
        outputFormat : int {PIL_FORMAT, NUMPY_FORMAT} (default : PIL_FORMAT)
            The output format to which the image will be converted
        """
        self._initImageBuffer(fileImageCollection, outputFormat)

        self._imgLoader = imageLoader

    def _extract(self, filename):
        return self._imgLoader.load(filename)


# #TODO add exceptions and stuff

# class ShufflerImageBuffer(ImageBuffer):

#     def __init__(self, imageBuffer, height, width, rate=0.5):
#         self._imgBuff = imageBuffer
#         self._shuffler = ss.ImageSuffler(height, width, rate)

#     def size(self):
#         return self._imgBuff.size()

#     def get(self, index):
#         img = self._imgBuff.get(index)
#         return self._shuffler.shuffle(img)

#     def get_labels(self):
#         return self._imgBuff.get_labels()
