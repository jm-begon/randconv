# -*- coding: utf-8 -*-
"""
A class for LearningSet
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__version__ = 'dev'

from ..util.indexer import Sliceable

class LearningSet(Sliceable):
    """
    ===========
    LearningSet
    ===========
    A :class:`LearningSet` is a buffer of predefined length which returns
    pairs of (image, label).

    The [int] operator returns a pair (image, label)
    The [int:int] operator returns an :class:`LearningSet`
    """
    def __init__(self, image_buffer, labels):
        if len(image_buffer) != len(labels):
            raise ValueError("Image buffer and labels must have the same length")
        self.image_buffer = image_buffer
        self.labels = labels


    def _get(self, index):
        return self.image_buffer[index], self.labels[index]

    def _slice(self, shallow_copy, slice_range):
        shallow_copy.image_buffer = self.image_buffer[slice_range]
        shallow_copy.labels = self.labels[slice_range]

    def __len__(self):
        return len(self.labels)

    def unzip(self):
        """
        Dissociate the images and the label. 

        Return
        ------
        (img_buffer, labels)
            img_buffer : :class:`ImageBuffer`
                An :class:`ImageBuffer` view of the images in the learning set
            labels : list of int
                The labels associated to the learning set
        """
        return self.image_buffer, self.labels
