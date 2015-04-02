# -*- coding: utf-8 -*-
"""
A set of factory function to help create usual cases of coordinator
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__date__ = "20 January 2015"

import math

from .image import *
from .util import (OddUniformGenerator, NumberGenerator,
                   CustomDiscreteNumberGenerator, GaussianNumberGenerator)
from .feature_extractor import ImageLinearizationExtractor, DepthCompressorILE
from .coordinator import (RandConvCoordinator, PyxitCoordinator)





class Const:
    RND_RU = "RND_RU"  # -1 (real uniform)
    RND_SET = "RND_SET"  # -2 (Discrete set with predifined probabilities)
    RND_GAUSS = "RND_GAUSS"  # (Gaussian distribution)

    FGEN_ORDERED = "FGEN_ORDERED"  # Ordered combination of others
    FGEN_CUSTOM = "FGEN_CUSTOM"  # Custom filters
    FGEN_ZEROPERT = "FGEN_ZEROPERT"  # Perturbation around origin
    FGEN_IDPERT = "FGEN_IDPERT"  # Perturbation around id filter
    FGEN_IDDIST = "FGEN_IDDIST"  # Maximum distance around id filter
    FGEN_STRAT = "FGEN_STRAT"  # Stratified scheme

    POOLING_NONE = "POOLING_NONE"  # 0
    POOLING_AGGREG_MIN = "POOLING_AGGREG_MIN"  # 1
    POOLING_AGGREG_AVG = "POOLING_AGGREG_AVG"  # 2
    POOLING_AGGREG_MAX = "POOLING_AGGREG_MAX"  # 3
    POOLING_CONV_MIN = "POOLING_MW_MIN"  # 4
    POOLING_CONV_AVG = "POOLING_MW_AVG"  # 5
    POOLING_CONV_MAX = "POOLING_MW_MAX"  # 6
    POOLING_MORPH_OPENING = "POOLING_MORPH_OPENING"  # 7
    POOLING_MORPH_CLOSING = "POOLING_MORPH_CLOSING"  # 8

    FEATEXT_ALL = "FEATEXTRACT_ALL"
    FEATEXT_SPASUB = "FEATEXTRACT_SPASUB"


def pyxit_factory(
        nb_subwindows=10,
        sw_min_size_ratio=0.5, sw_max_size_ratio=1.,
        sw_target_width=16, sw_target_height=16,
        fixed_size=False,
        sw_interpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        n_jobs=-1, verbosity=10, temp_folder=None,
        random=True):
    """
    Factory method to create :class:`PyxitCoordinator`

    Parameters
    ----------
    nb_subwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    sw_min_size_ratio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    sw_max_size_ratio : float : sw_min_size_ratio
    <= sw_max_size_ratio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    sw_target_width : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    sw_target_height : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    fixed_size : boolean (default : False)
        Whether to use fixe size subwindow. If False, subwindows are drawn
        randomly. If True, the target size is use as the subwindow size and
        only the position is drawn randomly
    sw_interpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    n_jobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    temp_folder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)
    random : bool (default : True)
        Whether to use randomness or use a predefined seed

    Return
    ------
        coordinator : :class:`Coordinator`
            The PyxitCoordinator (possibly decorated) corresponding to the set
            of parameters
    Notes
    -----
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """

    swngSeed = 0
    #Randomness
    if random:
        swngSeed = None
    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    if fixed_size:
        sw_extractor = FixTargetSWExtractor(sw_target_width,
                                           sw_target_height,
                                           sw_interpolation,
                                           swNumGenerator)
    else:
        sw_extractor = SubWindowExtractor(sw_min_size_ratio,
                                         sw_max_size_ratio,
                                         sw_target_width,
                                         sw_target_height,
                                         sw_interpolation,
                                         swNumGenerator)

    multi_sw_extractor = MultiSWExtractor(sw_extractor, nb_subwindows, True)

    #FEATURE EXTRACTOR
    feature_extractor = ImageLinearizationExtractor()

    #LOGGER
    autoFlush = verbosity >= 45
    logger = ProgressLogger(StandardLogger(autoFlush=autoFlush,
                                           verbosity=verbosity))

    #COORDINATOR
    coordinator = PyxitCoordinator(multi_sw_extractor, feature_extractor, logger,
                                   verbosity)

    if n_jobs != 1:
        coordinator.parallelize(n_jobs, temp_folder)
    return coordinator


def get_multi_poolers(poolings, finalHeight, finalWidth):
    #Aggregator
    poolers = []
    for height, width, policy in poolings:
        if policy is Const.POOLING_NONE:
            poolers.append(IdentityPooler())
        elif policy is Const.POOLING_AGGREG_AVG:
            poolers.append(AverageAggregator(width, height,
                                             finalWidth,
                                             finalHeight))
        elif policy is Const.POOLING_AGGREG_MAX:
            poolers.append(MaximumAggregator(width, height,
                                             finalWidth,
                                             finalHeight))
        elif policy is Const.POOLING_AGGREG_MIN:
            poolers.append(MinimumAggregator(width, height,
                                             finalWidth,
                                             finalHeight))
        elif policy is Const.POOLING_CONV_MIN:
            poolers.append(FastMWMinPooler(height, width))
        elif policy is Const.POOLING_CONV_AVG:
            poolers.append(FastMWAvgPooler(height, width))
        elif policy is Const.POOLING_CONV_MAX:
            poolers.append(FastMWMaxPooler(height, width))
        elif policy is Const.POOLING_MORPH_OPENING:
            poolers.append(MorphOpeningPooler(height, width))
        elif policy is Const.POOLING_MORPH_CLOSING:
            poolers.append(MorphClosingPooler(height, width))

    return MultiPooler(poolers)


def get_number_generator(genType, min_value, max_value, seed, **kwargs):
    if genType is Const.RND_RU:
        value_generatorerator = NumberGenerator(min_value, max_value, seed)
    elif genType is Const.RND_SET:
        probLaw = kwargs["probLaw"]
        value_generatorerator = CustomDiscreteNumberGenerator(probLaw, seed)
    elif genType is Const.RND_GAUSS:
        if "outRange" in kwargs:
            outRange = kwargs["outRange"]
            value_generatorerator = GaussianNumberGenerator(min_value, max_value, seed,
                                                   outRange)
        else:
            value_generatorerator = GaussianNumberGenerator(min_value, max_value, seed)
    return value_generatorerator


def get_filter_generator(policy, parameters, nb_filterss, random=False):
    if policy == Const.FGEN_ORDERED:
        #Parameters is a list of tuples (policy, parameters)
        ls = []
        subNbFilters = int(math.ceil(nb_filterss/len(parameters)))

        for subPolicy, subParameters in parameters:
            ls.append(get_filter_generator(subPolicy, subParameters,
                                         subNbFilters, random))
        return OrderedMFF(ls, nb_filterss)

    if policy is Const.FGEN_CUSTOM:
        print "Custom filters"
        return custom_finite_3_same_filter()

    #Parameters is a dictionary
    valSeed = None
    sizeSeed = None
    shuffling_seed = None
    perturbationSeed = None
    cell_seed = None
    sparseSeed = 5
    if random:
        valSeed = 1
        sizeSeed = 2
        shuffling_seed = 3
        perturbationSeed = 4
        cell_seed = 5
        sparseSeed = 6

    min_size = parameters["min_size"]
    max_size = parameters["max_size"]
    size_generatorerator = OddUniformGenerator(min_size, max_size, seed=sizeSeed)

    min_val = parameters["min_val"]
    max_val = parameters["max_val"]
    value_generator = parameters["value_generator"]
    value_generatorerator = get_number_generator(value_generator, min_val, max_val,
                                      valSeed, **parameters)

    normalization = None
    if "normalization" in parameters:
        normalization = parameters["normalization"]

    if policy is Const.FGEN_ZEROPERT:
        print "Zero perturbation filters"
        baseFilterGenerator = FilterGenerator(value_generatorerator, size_generatorerator,
                                              normalisation=normalization)

    elif policy is Const.FGEN_IDPERT:
        print "Id perturbation filters"
        baseFilterGenerator = IdPerturbatedFG(value_generatorerator, size_generatorerator,
                                              normalisation=normalization)
    elif policy is Const.FGEN_IDDIST:
        print "Id distance filters"
        max_dist = parameters["max_dist"]
        baseFilterGenerator = IdMaxL1DistPerturbFG(value_generatorerator, size_generatorerator,
                                                   max_dist,
                                                   normalisation=normalization,
                                                   shuffling_seed=shuffling_seed)
    elif policy is Const.FGEN_STRAT:
        print "Stratified filters"
        nb_cells = parameters["strat_nb_cells"]
        minPerturbation = 0
        if "minPerturbation" in parameters:
            minPerturbation = parameters["minPerturbation"]
        maxPerturbation = 1
        if "maxPerturbation" in parameters:
            maxPerturbation = parameters["maxPerturbation"]
        perturbationGenerator = get_number_generator(value_generator,
                                                   minPerturbation,
                                                   maxPerturbation,
                                                   perturbationSeed)
        baseFilterGenerator = StratifiedFG(min_val, max_val, nb_cells,
                                           perturbationGenerator,
                                           size_generatorerator,
                                           normalisation=normalization,
                                           cell_seed=cell_seed)

    if "sparse_proba" in parameters:
        print "Adding sparcity"
        sparse_proba = parameters["sparse_proba"]
        baseFilterGenerator = SparsityDecoratorFG(baseFilterGenerator,
                                                  sparse_proba,
                                                  sparseSeed)

    print "Returning filters"
    return Finite3SameFilter(baseFilterGenerator, nb_filterss)


def get_feature_extractor(policy, **kwargs):
    if policy is Const.FEATEXT_SPASUB:
        nbCol = kwargs.get("nbCol", 2)
        return DepthCompressorILE(nbCol)

    else:  # Suupose Const.FEATEXT_ALL
        return ImageLinearizationExtractor()

#TODO : include in randconv : (Const.FEATEXT_ALL, {}), (Const.FEATEXT_SPASUB, {"nbCol":2})
def randconv_factory(
        nb_filterss=5,
        filter_policy=(Const.FGEN_ZEROPERT,
                       {"min_size": 2, "max_size": 32, "min_val": -1, "max_val": 1,
                        "value_generator": Const.RND_RU,
                        "normalization": FilterGenerator.NORMALISATION_MEANVAR}),
        poolings=[(3, 3, Const.POOLING_AGGREG_AVG)],
        extractor=(Const.FEATEXT_ALL, {}),
        nb_subwindows=10,
        sw_min_size_ratio=0.5, sw_max_size_ratio=1.,
        sw_target_width=16, sw_target_height=16,
        sw_interpolation=SubWindowExtractor.INTERPOLATION_BILINEAR,
        include_original_img=False,
        n_jobs=-1, verbosity=10, temp_folder=None,
        random=True):
    """
    Factory method to create :class:`RandConvCoordinator` tuned for RGB images

    Parameters
    ----------
    nb_filterss : int >= 0 (default : 5)
        The number of filter

    filter_policy : pair (policyType, parameters)
        policyType : one of Const.FGEN_*
            The type of filter generation policy to use
        parameters : dict
            The parameter dictionnary to forward to :func:`get_filter_generator`

    poolings : iterable of triple (height, width, policy) (default :
    [(3, 3, Const.POOLING_AGGREG_AVG)])
        A list of parameters to instanciate the according :class:`Pooler`
        height : int > 0
            the height of the neighborhood window
        width : int > 0
            the width of the neighborhood window
        policy : int in {Const.POOLING_NONE, Const.POOLING_AGGREG_MIN,
    Const.POOLING_AGGREG_AVG, Const.POOLING_AGGREG_MAX,
    Const.POOLING_CONV_MIN, Const.POOLING_CONV_AVG, Const.POOLING_CONV_MAX}

    nb_subwindows : int >= 0 (default : 10)
        The number of subwindow to extract
    sw_min_size_ratio : float > 0 (default : 0.5)
        The minimum size of a subwindow expressed as the ratio of the size
        of the original image
    sw_max_size_ratio : float : sw_min_size_ratio
    <= sw_max_size_ratio <= 1 (default : 1.)
        The maximim size of a subwindow expressed as the ratio of the size
        of the original image
    sw_target_width : int > 0 (default : 16)
        The width of the subwindows after reinterpolation
    sw_target_height : int > 0 (default : 16)
        The height of the subwindows after reinterpolation
    sw_interpolation : int (default :
    SubWindowExtractor.INTERPOLATION_BILINEAR)
        The subwindow reinterpolation algorithm. For more information, see
        :class:`SubWindowExtractor`

    include_original_img : boolean (default : False)
        Whether or not to include the original image in the subwindow
        extraction process

    n_jobs : int >0 or -1 (default : -1)
        The number of process to spawn for parallelizing the computation.
        If -1, the maximum number is selected. See also :mod:`Joblib`.
    verbosity : int >= 0 (default : 10)
        The verbosity level
    temp_folder : string (directory path) (default : None)
            The temporary folder used for memmap. If none, some default folder
            will be use (see the :class:`ParallelCoordinator`)

    random : bool (default : True)
        Whether to use randomness or use a predefined seed

    Return
    ------
        coordinator : :class:`Coordinator`
            The RandConvCoordinator corresponding to the
            set of parameters

    Notes
    -----
    - Filter generator
        Base instance of :class:`Finite3SameFilter` with a base instance of
        :class:`NumberGenerator` for the values and
        :class:`OddUniformGenerator` for the sizes
    - Filter size
        The filter are square (same width as height)
    - Convolver
        Base instance of :class:`RGBConvolver`
    - Subwindow random generator
        The subwindow random generator is a :class:`NumberGenerator` base
        instance (generate real nubers uniformely).
    - Feature extractor
        Base instance of :class:`ImageLinearizationExtractor`
    """
    #RANDOMNESS
    swngSeed = None
    if random is False:
        swngSeed = 0

    #CONVOLUTIONAL EXTRACTOR
    #Filter generator
    #Type/policy parameters, #filters, random
    filter_policyType, filter_policyParam = filter_policy
    filter_generator = get_filter_generator(filter_policyType, filter_policyParam,
                                         nb_filterss, random)

    #Convolver
    convolver = RGBConvolver()

    #Aggregator
    multi_pooler = get_multi_poolers(poolings, sw_target_height,
                                  sw_target_width)

    #SubWindowExtractor
    swNumGenerator = NumberGenerator(seed=swngSeed)
    sw_extractor = SubWindowExtractor(sw_min_size_ratio,
                                     sw_max_size_ratio,
                                     sw_target_width,
                                     sw_target_height,
                                     sw_interpolation, swNumGenerator)

    multi_sw_extractor = MultiSWExtractor(sw_extractor, nb_subwindows, False)

    #ConvolutionalExtractor
    convolutional_extractor = ConvolutionalExtractor(filter_generator,
                                                     convolver,
                                                     multi_sw_extractor,
                                                     multi_pooler,
                                                     include_original_img)
    #FEATURE EXTRACTOR
    feature_extractor = get_feature_extractor(extractor[0], **extractor[1])


    #COORDINATOR
    coordinator = RandConvCoordinator(convolutional_extractor, feature_extractor)

    if n_jobs != 1:
        coordinator.parallelize(n_jobs, temp_folder)
    return coordinator

