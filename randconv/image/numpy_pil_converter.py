# -*- coding: utf-8 -*-
# Author : Jean-Michel Begon
# Date : Mar 08 2014
"""
A convertor between the two image format we use : PIL and numpy
See :mod:`ImageBuffer` for more information
"""
try:
    import Image
except ImportError:
    from PIL import Image
import numpy as np

__all__ = ["NumpyPILConvertor", "NumpyToPILConvertor", "PILToNumpyConvertor"]


class NumpyPILConvertor:
    """
    ==================
    NumpyPILConvertor
    ==================
    A convertor between PIL image and numpy
    """

    def numpy2pil(self, np_image):
        """
        Convert a numpy image into a PIL image
        Parameters
        ----------
        np_image : a numpy image
            The image to convert
        Return
        ------
        pilImage : a :class:`PIL.Image`
            The converted image
        """
        if isinstance(np_image, Image.Image):
            return np_image
        return Image.fromarray(np.uint8(np_image.clip(0, 255)))

    def pil2numpy(self, pil_img):
        """
        Convert a a PIL image into numpy image
        Parameters
        ----------
        pilImage : a :class:`PIL.Image`
            The image to convert
        Return
        ------
        np_image : a numpy image
            The converted image
        """
        if isinstance(pil_img, np.ndarray):
            return pil_img
        return np.array(pil_img)


class NumpyToPILConvertor(NumpyPILConvertor):
    """
    ===================
    NumpyToPILConvertor
    ===================
    A numpy to PIL image convertor
    """
    def __call__(self, img):
        """Delegates to :meth:`numpyToPil`"""
        return self.numpy2pil(img)


class PILToNumpyConvertor(NumpyPILConvertor):
    """
    ===================
    PILToNumpyConvertor
    ===================
    A PIL image to numpy convertor
    """
    def __call__(self, img):
        """Delegates to :meth:`pil2numpy`"""
        return self.pil2numpy(img)
