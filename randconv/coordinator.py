# -*- coding: utf-8 -*-
"""
:class:`Coordinator` are responsible for applying a feature extraction
mechanism to all the data contained in a image_buffer and keeping the
consistency if it creates several feature vectors for one image
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__date__ = "20 January 2015"

from abc import ABCMeta, abstractmethod
import numpy as np

from progressmonitor import monitor_with

from .util import SerialExecutor, ParallelExecutor
from .image import Rescaler, MaxoutRescaler
from .image import NumpyPILConvertor




class Coordinator(object):
    """
    ===========
    Coordinator
    ===========

    :class:`Coordinator` are responsible for applying a feature extraction
    mechanism to all the data contained in a image_buffer and keeping the
    consistency if it creates several feature vectors for one image.

    The extraction mechanism is class dependent. It is the class
    responsability to document its policy.

    """

    __metaclass__ = ABCMeta

    def __init__(self, dtype=np.float32, label_type=np.uint8):
        self._exec = SerialExecutor()
        self._dtype = dtype
        self._label_type = label_type
        if dtype is np.float:
            self._rescaler = Rescaler()
        else:
            self._rescaler = MaxoutRescaler(dtype)

    def parallelize(self, n_jobs=-1, temp_folder=None):
        """
        Parallelize the coordinator

        Parameters
        ----------
        n_jobs : int {>0, -1} (default : -1)
            The parallelization factor. If "-1", the maximum factor is used
        temp_folder : filepath (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :lib:`joblib` library)
        """
        #TODO manager verbosity
        self._exec = ParallelExecutor(n_jobs, 50, temp_folder)

    @monitor_with("func.rc.Coord", task_name="Processing images")
    def process(self, image_buffer, learning_phase=True):
        """
        Extracts the feature vectors for the images contained in the
        :class:`ImageBuffer`

        Abstract method to overload.

        Parameters
        ----------
        image_buffer : :class:`ImageBuffer`
            The data to process
        learning_phase : bool (default : True)
            Specifies whether it is the learning phase. For some
            :class:`Coordinator`, this is not important but it might be for
            the stateful ones

        Return
        ------
        X : a numpy 2D array
            the N x M feature matrix. Each of the N rows correspond to an
            object and each of the M columns correspond to a variable
        y : an iterable of int
            the N labels corresponding to the N objects of X

        Note
        ----
        The method might provide several feature vectors per original image.
        It ensures the consistency with the labels and is explicit about
        the mapping.

        Implementation
        --------------
        The method :meth:`process` only "schedule" the work. The
        implementation of what is to be done is the responbility of the method
        :meth:`_onProcess`. It is this method that should be overloaded
        """
        self._nbColors = image_buffer.nb_bands()


        nb_features = self.nb_features_per_object(self._nbColors)
        nbObjs = self.nb_objects(image_buffer)

        X = self._exec.create_array((nbObjs, nb_features), self._dtype)
        y = self._exec.create_array((nbObjs), self._label_type)

        self._exec.execute_with_index(self._onProcess, image_buffer,
                                      learning_phase=learning_phase,
                                      X_res=X, y_res=y)

        return X, y

    @abstractmethod
    def _onProcess(self, image_buffer, start_index, learning_phase,
                   X_res, y_res):
        """
        Extracts the feature vectors for the images contained in the
        :class:`ImageBuffer`

        Abstract method to overload.

        Parameters
        ----------
        image_buffer : :class:`ImageBuffer`
            The data to process
        learning_phase : bool (default : True)
            Specifies whether it is the learning phase. For some
            :class:`Coordinator`, this is not important but it might be for
            the stateful ones

        Return
        ------
        X : a numpy 2D array
            the N x M feature matrix. Each of the N rows correspond to an
            object and each of the M columns correspond to a variable
        y : an iterable of int
            the N labels corresponding to the N objects of X

        Note
        ----
        The method might provide several feature vectors per original image.
        It ensures the consistency with the labels and is explicit about
        the mapping.
        """
        pass

    def __call__(self, image_buffer, learning_phase):
        """Delegate to :meth:`process`"""
        return self.process(image_buffer, learning_phase)

    def clean(self, *args):
        for resource in args:
            self._exec.clean(resource)

    @abstractmethod
    def nb_features_per_object(self, nbColors=1):
        """
        Return the number of features that this :class:`Coordinator` will
        produce per object
        """
        pass

    def nb_objects(self, image_buffer):
        """
        Return the number of objects that this :class:`Coordinator` will
        produce
        """
        return len(image_buffer)



class PyxitCoordinator(Coordinator):
    """
    ================
    PyxitCoordinator
    ================

    This coordinator uses a :class:`MultiSWExtractor` and a
    :class:`FeatureExtractor`. The first component extracts subwindows
    from the image while the second extract the features from each subwindow.

    Thus, this method creates several feature vectors per image. The number
    depends on the :class:`MultiSWExtractor` instance but are grouped
    contiguously.

    Note
    ----
    The :class:`FeatureExtractor` instance must be adequate wrt the image
    type
    """
    def __init__(self, multi_sw_extractor, feature_extractor, logger=None,
                 verbosity=None):
        """
        Construct a :class:`PyxitCoordinator`

        Parameters
        ----------
        multi_sw_extractor : :class:`MultiSWExtractor`
            The component responsible for the extraction of subwindows
        feature_extractor: :class:`FeatureExtractor`
            The component responsible for extracting the features from
            each subwindow
        """
        Coordinator.__init__(self, logger, verbosity)
        self._multi_sw_extractor = multi_sw_extractor
        self._feature_extractor = feature_extractor

    def _onProcess(self, image_buffer, start_index, learning_phase,
                   X_res, y_res):
        """Overload"""

        convertor = NumpyPILConvertor()

        #Init
        index = start_index * self.nb_obj_multiplicator()

        #Main loop
        for image, label in monitor_with("gen.rc.Coord.Pyxit")(image_buffer):
            image = convertor.numpy2pil(image)
            imgLs = self._multi_sw_extractor.extract(image)
            #Filling the X and y
            for img in imgLs:
                tmpRes = self._feature_extractor.extract(
                    convertor.pil2numpy(img))
                X_res[index] = self._rescaler(tmpRes)
                y_res[index] = label
                index += 1

    def nb_features_per_object(self, nbColors):
        height, width = self._multi_sw_extractor.get_final_size()
        return self._feature_extractor.nb_features_per_object(height,
                                                          width,
                                                          nbColors)

    def nb_obj_multiplicator(self):
        return self._multi_sw_extractor.nb_subwidows()

    def nb_objects(self, image_buffer):
        return len(image_buffer)*self.nb_obj_multiplicator()


class RandConvCoordinator(Coordinator):
    """
    ===================
    RandConvCoordinator
    ===================

    This coordinator uses a :class:`ConvolutionalExtractor` and a
    :class:`FeatureExtractor`. The first component extracts subwindows from
    the image applies filter to each subwindow and aggregate them while the
    second extract the features from each subwindow.

    Thus, this method creates several feature vectors per image. The number
    depends on the :class:`ConvolutionalExtractor` instance but are grouped
    contiguously.


    Constructor parameters
    ----------------------
    convolutional_extractor : :class:`ConvolutionalExtractor`
        The component responsible for the extraction, filtering and
        aggregation of subwindows
    feature_extractor: :class:`FeatureExtractor`
        The component responsible for extracting the features from
        each filtered and aggregated subwindow

    Note
    ----
    The :class:`FeatureExtractor` instance must be adequate wrt the image
    type
    """

    def __init__(self, convolutional_extractor, feature_extractor,
                 dtype=np.float32, label_type=np.uint8):

        Coordinator.__init__(self, dtype, label_type)
        self._convol_extractor = convolutional_extractor
        self._feature_extractor = feature_extractor

    def _onProcess(self, image_buffer, start_index, learning_phase,
                   X_res, y_res):
        """Overload"""

        #Init
        row = start_index * self.nb_obj_multiplicator()

        #Main loop
        for image, label in monitor_with("gen.rc.Coord.RC")(image_buffer):
            #Get the subwindows x filters
            all_subwindows = self._convol_extractor.extract(image)

            #Accessing each subwindow set separately
            for filtered_ls in all_subwindows:
                #Accessing each filter separately for a given subwindow
                column = 0
                for filtered in filtered_ls:

                    #Extracting the features for each filter
                    filter_feature = self._feature_extractor.extract(filtered)
                    X_res[row, column:(column+len(filter_feature))] = filter_feature
                    column += len(filter_feature)

                #Corresponding label
                y_res[row] = label
                row += 1


    def get_filters(self):
        """
        Return the filters used to process the image

        Return
        ------
        filters : iterable of numpy arrays
            The filters used to process the image, with the exclusion
            of the identity filter if the raw image was included
        """
        return self._convol_extractor.get_filters()

    def is_image_included(self):
        """
        Whether the raw image was included

        Return
        ------
        isIncluded : boolean
            True if the raw image was included
        """
        return self._convol_extractor.is_image_included()

    def _groups_info(self, nb_features):
        """
        Return information about the grouping of features (original image
        included if necessary)

        Parameters
        ----------
         nb_features : int > 0
            The number of features

        Return
        ------
        tuple = (nb_features, nb_groups, nb_feature_per_group)
        nb_features : int
            The number of features
        nb_groups : int
            The number of groups
        nb_feature_per_group : int
            The number of features per group
        """
        nb_groups = len(self.get_filters())*len(self._convol_extractor.get_poolers())
        if self.is_image_included():
            nb_groups += len(self._convol_extractor.get_poolers())
        nb_feature_per_group = nb_features // nb_groups
        return nb_features, nb_groups, nb_feature_per_group

    def feature_groups(self, nb_features):
        """
        Returns an iterable of start indices of the feature groups of X and
        the number of features

        Parameters
        ----------
        nb_features : int > 0
            The number of features

        Return
        ------
        tuple = (nb_features, nb_groups, ls)
        nb_features : int
            The number of features
        nb_groups : int
            The number of groups
        ls : iterable of int
            Returns an iterable of start indices of the feature groups of X
            and the number of features
        """
        nb_features, nb_groups, nb_feature_per_group = self._groups_info(nb_features)
        return (nb_features, nb_groups, xrange(0, nb_features+1,
                                             nb_feature_per_group))

    def importance_per_feature_grp(self, classifier):
        """
        Computes the importance of each filter.

        Parameters
        ----------
        classifier : sklearn.ensemble classifier with
        :attr:`feature_importances_`
            The classifier (just) used to fit the model
        X : 2D numpy array
            The feature array. It must have been learnt by this
            :class:`ConvolutionalExtractor` with the given classifier
        Return
        ------
        pair = (importance, indices)
        importance : iterable of real
            the importance of each group of feature
        indices : iterable of int
            the sorted indices of importance in decreasing order
        """

        importance = classifier.feature_importances_
        nb_features, nb_groups, starts = self.feature_groups(len(importance))
        img_per_grp = []
        for i in xrange(nb_groups):
            img_per_grp.append(sum(importance[starts[i]:starts[i+1]]))

        return img_per_grp, np.argsort(img_per_grp)[::-1]

    def nb_features_per_object(self, nbColors):
        # Number of filters * poolers
        nb_groups = len(self.get_filters())*len(self._convol_extractor.get_poolers())
        if self.is_image_included():
            nb_groups += len(self._convol_extractor.get_poolers())

        #Size of the extracted subwindows
        height, width = self._convol_extractor.get_final_size_per_subwindow()

        #Number of features per individual subwindows
        nbFperGroup = self._feature_extractor.nb_features_per_object(height,
                                                                 width,
                                                                 nbColors)
        #Total number of features
        return nb_groups*nbFperGroup

    def nb_obj_multiplicator(self):
        return self._convol_extractor.get_nb_subwindows()

    def nb_objects(self, image_buffer):
        return len(image_buffer)*self.nb_obj_multiplicator()



