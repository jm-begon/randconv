# -*- coding: utf-8 -*-

""" """

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__date__ = "20 January 2015"

import numpy as np
import scipy.sparse as sps

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomTreesEmbedding
from progressmonitor import monitor_with

from ._compute_histogram import compute_histogram



class Classifier(object):
    """
    ==========
    Classifier
    ==========
    A :class:`Classifier` uses a :class:`Coordinator` to extract data from
    an :class:`ImageBuffer` and feed it to a **scikit-learn base classifier**.
    The :class:`Classifier` can take care of multiple feature vectors per
    object.
    """
    def __init__(self, coordinator, base_classifier):
        """
        Construct a :class:`Classifier`

        Parameters
        ----------
        coordinator : :class:`Coordinator`
            The coordinator responsible for the features extraction
        base_classifier : scikit-learn classifier (:meth:`predict_proba`
        required)
            The learning algorithm which will classify the data
        """
        self._classifier = base_classifier
        self._coord = coordinator
        self._classif2user_lut = []
        self._user2classif_lut = {}

    def _build_luts(self, y_user):
        """
        Builds the lookup tables for converting user labels to/from
        classifier label

        Parameters
        ----------
        y_user : list
           the list of user labels
        """
        user_labels = np.unique(y_user)
        self._classif2user_lut = user_labels
        self._user2classif_lut = {j: i for i, j in enumerate(user_labels)}

    def _convert_labels(self, y_user):
        """
        Convert labels from the user labels to the internal labels
        Parameters
        ----------
        y_user : list
           list of user labels to convert into internal labels
        Returns
        -------
        y_classif : list
           the corresponding internal labels
        """
        print self._user2classif_lut
        return [self._user2classif_lut[x] for x in y_user]

    def _convert_labels_back(self, y_classif):
        """
        Convert labels back to the user labels
        Parameters
        ----------
        y_classif : list
           list of internal labels to convert
        Returns
        -------
        y_user : list
           the corresponding user labels
        """
        return [self._classif2user_lut[x] for x in y_classif]

    @monitor_with("rc.func.RCClassifier", task_name="Learning the model")
    def fit(self, image_buffer):
        """
        Fits the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to learn from

        Return
        -------
        self : :class:`Classifier`
            This instance
        """
        #Updating the labels
        y_user = image_buffer.get_labels()
        self._build_luts(y_user)

        #Extracting the features
        X, y_user = self._coord.process(image_buffer, learning_phase=True)

        #Converting the labels
        y = self._convert_labels(y_user)

        #Delegating the classification
        self._classifier.fit(X, y)

        #Cleaning up
        self._coord.clean(X, y_user)

        return self

    def predict_predict_proba(self, image_buffer):
        """
        Classify the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        pair : (y_proba, y_classif)
        y_proba: list of list of float
            each entry is the probability vector of the input of the same
            index as computed by the base classifier
        y_classif : a list of int
            each entry is the classification label corresponding to the input
        """
        y_prob = self.predict_proba(image_buffer)
        y_classif = np.argmax(y_prob, axis=1)
        return y_prob, self._convert_labels_back(y_classif)

    @monitor_with("func.rc.Classifier", task_name="Classifying")
    def predict(self, image_buffer):
        """
        Classify the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        list : list of int
            each entry is the classification label corresponding to the input
        """
        _, y_classif = self.predict_predict_proba(image_buffer)
        return y_classif

    @monitor_with("func.rc.Classifier", task_name="Classifying")
    def predict_proba(self, image_buffer):
        """
        Classify softly the data contained is the :class:`ImageBuffer`
        instance. i.e. yields a probability vector of belongin to each
        class

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        list : list of list of float
            each entry is the probability vector of the input of the same
            index as computed by the base classifier
        """
        #Extracting the features
        X_pred, y = self._coord.process(image_buffer, learning_phase=False)

        #Cleaning up
        self._coord.clean(y)
        del y

        y = self._predict_proba(X_pred, len(image_buffer))

        #Cleaning up
        self._coord.clean(X_pred)
        del X_pred

        return y

    def _predict_proba(self, X_pred, nb_objects):
        #Misc.
        nb_factor = len(X_pred)/nb_objects

        y = np.zeros((nb_objects, len(self._user2classif_lut)))

        #Classifying the data
        _y = self._classifier.predict_proba(X_pred)

        for i in xrange(nb_objects):
            y[i] = np.sum(_y[i * nb_factor:(i+1) * nb_factor], axis=0) / nb_factor

        return y

    def _predict(self, X_pred, nb_objects):
        y_classif = np.argmax(self._predict_proba(X_pred, nb_objects), axis=1)
        return self._convert_labels_back(y_classif)

    def accuracy(self, y_pred, y_truth):
        """
        Compute the frequency of correspondance between the two vectors

        Parameters
        -----------
        y_pred : list of int
            The prediction by the model
        y_truth : list of int
            The ground truth

        Return
        -------
        accuracy : float
            the accuracy
        """
        s = sum([1 for x, y in zip(y_pred, y_truth) if x == y])
        return float(s)/len(y_truth)


    def confusion_matrix(self, y_pred, y_truth):
        """
        Compute the confusion matrix

        Parameters
        -----------
        y_pred : list of int
            The prediction by the model
        y_truth : list of int
            The ground truth

        Return
        -------
        mat : 2D numpy array
            The confusion matrix
        """
        return confusion_matrix(y_truth, y_pred)


class UnsupervisedVisualBagClassifier(Classifier):
    """
    ===============================
    UnsupervisedVisualBagClassifier
    ===============================
    1. Unsupervised
    2. Binary bag of words
    3. Totally random trees
    """

    def __init__(self, coordinator, base_classifier, n_estimators=10,
                 max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 n_jobs=-1, random_state=None, verbose=0):
        Classifier.__init__(self, coordinator, base_classifier)
        self.histoSize = 0
        self._visualBagger = RandomTreesEmbedding(n_estimators=n_estimators,
                                                  max_depth=max_depth,
                                                  min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf,
                                                  n_jobs=n_jobs,
                                                  random_state=random_state,
                                                  verbose=verbose)

    @monitor_with("func.rc.UVBClassif", task_name="Extracting features")
    def _preprocess(self, image_buffer, learning_phase):


        X_pred, y = self._coord.process(image_buffer,
                                        learning_phase=learning_phase)

        y_user = self._convert_labels(y)

        #Cleaning up
        self._coord.clean(y)
        del y


        #Bag-of-word transformation
        with monitor_with("code.rc.UVBClassif", taskname="Bag-of-word transformation"):

            X2 = None
            if learning_phase:
                X2 = self._visualBagger.fit_transform(X_pred, y_user)
                self.histoSize = X2.shape[1]
            else:
                X2 = self._visualBagger.transform(X_pred)

            #Cleaning up
            self._coord.clean(X_pred)
            del X_pred
            del y_user


        nb_factor = X2.shape[0] // len(image_buffer)

        if not sps.isspmatrix_csr(X2):
            X2 = X2.tocsr()

        if nb_factor == 1:
            return X2

        with monitor_with("code.rc.UVBClassif", taskname="Histogram"):
            nbTrees = self._visualBagger.n_estimators
            X3 = compute_histogram(len(image_buffer), nb_factor, nbTrees, X2)

            #Cleaning up
            del X2  # Should be useless

        return X3

    @monitor_with("func.rc.UVBClassif", task_name="Fitting histogram")
    def fit_histogram(self, hist, y):
        #Delegating the classification
        self._classifier.fit(hist, y)
        return self

    def fit(self, image_buffer):
        """
        Fits the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to learn from

        Return
        -------
        self : :class:`Classifier`
            This instance
        """
        #Updating the labels
        y_user = image_buffer.get_labels()
        self._build_luts(y_user)
        y = self._convert_labels(y_user)

        X = self._preprocess(image_buffer, learning_phase=True)

        return self.fit_histogram(X, y)

    def predict(self, image_buffer):
        """
        Classify the data contained in the :class:`ImageBuffer` instance

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        list : list of int
            each entry is the classification label corresponding to the input
        """

        X = self._preprocess(image_buffer, learning_phase=False)
        y_classif = self._classifier.predict(X)
        return self._convert_labels_back(y_classif)

    def predict_proba(self, image_buffer):
        """
        Classify softly the data contained is the :class:`ImageBuffer`
        instance. i.e. yields a probability vector of belongin to each
        class

        Parameters
        -----------
        image_buffer : :class:`ImageBuffer`
            The data to classify

        Return
        -------
        list : list of list of float
            each entry is the probability vector of the input of the same
            index as computed by the base classifier
        """
        if not hasattr(self._classifier, "predict_proba"):
            #Early error
            self._classifier.predict_proba(np.zeros((1, 1)))

        X = self._preprocess(image_buffer, learning_phase=False)
        return self._classifier.predict_proba(X)
