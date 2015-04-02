# -*- coding: utf-8 -*-
"""
A set of subwindow extractor
"""
__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "3-clause BSD License"
__date__ = "20 January 2015"

try:
    import Image
except ImportError:
    from PIL import Image
from skimage.transform import resize

__all__ = ["SubWindowExtractor", "FixTargetSWExtractor",
           "FixImgSWExtractor", "MultiSWExtractor"]


class SubWindowExtractor:
    """
    ==================
    SubWindowExtractor
    ==================
    A :class:`SubWindowExtractor` extract subwindows from an image and resize
    them to a given shape.

    The size and location of a given subwindow are drawn randomly

    Interpolation
    -------------
    The resizing step needs a interpolation algorithm :
    INTERPOLATION_NEAREST
        Nearest neighbor interpolation
    INTERPOLATION_BILINEAR
        bilinear interpolation
    INTERPOLATION_CUBIC
        bicubic interpolation
    INTERPOLATION_ANTIALIAS
        antialisaing interpolation
    """
    INTERPOLATION_NEAREST = 1
    INTERPOLATION_BILINEAR = 2
    INTERPOLATION_CUBIC = 3
    INTERPOLATION_ANTIALIAS = 4

    def __init__(self, min_size, max_size, target_width, target_height,
                 interpolation, number_generator):
        """
        Construct a :class:`SubWindowExtractor`

        Parameters
        ----------
        min_size : float 0 < min_size <= 1
            The minimum size of subwindow express as the size ratio with
            the original image
        max_size : float min_size <= max_size <= 1
            The maximum size of subwindow express as the size ratio with
            the original image
        target_width : int > 0
            The width of the subwindow after resizing
        target_height : int > 0
            the height of the subwindow after resizing
        interpolation : int {INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR,
        INTERPOLATION_CUBIC, INTERPOLATION_ANTIALIAS}
            The reintorpolation mechanism
        number_generator : :class:`NumberGenerator`
            The random number generator used for drawing the subwindows. It
            draws the height and width of the subwindow (respecting the
            original ratio) and then draws the location. Real number generator
            are fine and will be casted into int
        """
        self._min_size = min_size
        self._max_size = max_size
        self._target_width = target_width
        self._target_height = target_height
        self._numGen = number_generator
        self.set_interpolation(interpolation)

    def set_interpolation(self, interpolation):
        """
        Set the interpolation algorithm for this :class:`SubWindowExtractor`
        instance.

        Paramters
        ---------
        interpolation : int {INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR,
        INTERPOLATION_CUBIC, INTERPOLATION_ANTIALIAS}
            The reintorpolation mechanism
        """
        if interpolation == SubWindowExtractor.INTERPOLATION_NEAREST:
            pil_interpolation = Image.NEAREST
        elif interpolation == SubWindowExtractor.INTERPOLATION_BILINEAR:
            pil_interpolation = Image.BILINEAR
        elif interpolation == SubWindowExtractor.INTERPOLATION_CUBIC:
            pil_interpolation = Image.CUBIC
        elif interpolation == SubWindowExtractor.INTERPOLATION_ANTIALIAS:
            pil_interpolation = Image.ANTIALIAS
        else:
            pil_interpolation = Image.BILINEAR
        self._interpolation = pil_interpolation

    def get_crop_box(self, width, height):
        """
        Draws a new crop box

        Paramters
        ---------
        width : int > 0
            the width of the image on which the cropbox will be used
        height: int > 0
            the height of the image on which the cropbox will be used

        Return
        ------
        tuple = (sx, sy, ex, ey)
        sx : int
            The x-coordinate of the upper left pixel of the cropbox
        sy : int
            The y-coordinate of the upper left pixel of the cropbox
        ex : int
            the x-coordinate of the lower right pixel of the cropbox
        ey : int
            the y-coordinate of the lower right pixel of the cropbox
        """
        if width < height:
            ratio = 1. * self._target_height / self._target_width
            min_width = self._min_size * width
            max_width = self._max_size * width

            if min_width * ratio > height:
                raise ValueError

            if max_width * ratio > height:
                max_width = height / ratio

            crop_width = self._numGen.get_number(min_width, max_width)
            crop_height = ratio * crop_width

        else:
            ratio = 1. * self._target_width / self._target_height
            min_height = self._min_size * height
            max_height = self._max_size * height

            if min_height * ratio > width:
                raise ValueError

            if max_height * ratio > width:
                max_height = width / ratio

            crop_height = self._numGen.get_number(min_height, max_height)
            crop_width = ratio * crop_height

        if crop_width == 0:
            crop_width = 1
        if crop_height == 0:
            crop_height = 1

        # Draw a random position
        sx = int(self._numGen.get_number(0, width-crop_width))
        sy = int(self._numGen.get_number(0, height-crop_height))

        # Crop subwindow
        return (sx, sy, int(sx+crop_width), int(sy+crop_height))

    def extract_with_boxes(self, image):
        """
        Extract a subwindow of an image

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindow

        Return
        ------
        pair = (subwindow, box)
        subwindow : PIL.Image
            The subwindow extracted from the original image
        box = (sx, sy, ex, ey) the croping box
        sx : int
            The x-coordinate of the upper left pixel of the cropbox
        sy : int
            The y-coordinate of the upper left pixel of the cropbox
        ex : int
            the x-coordinate of the lower right pixel of the cropbox
        ey : int
            the y-coordinate of the lower right pixel of the cropbox
        """
        # Draw a random window
        width, height = image.size
        try:
            box = self.get_crop_box(width, height)
        except CorpLargerError:
            #subwindow larger than image, so we simply resize original image
            #to target sizes
            sub_window = image.resize((self._target_width, self._target_height),
                                     self._interpolation)
            return sub_window, box

        return self.crop_and_resize(image, box)

    def extract(self, image):
        """
        Extract a subwindow of an image

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindow

        Return
        ------
        subwindow : PIL.Image
            The subwindow extracted from the original image
        """
        sw, box = self.extract_with_boxes(image)
        return sw

    def crop_and_resize(self, image, cropbox):
        """
        Apply image cropping and resize image thanks to the instance
        reinterpolation mechanism

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindow
        box = (sx, sy, ex, ey) the croping box
            sx : int
                The x-coordinate of the upper left pixel of the cropbox
            sy : int
                The y-coordinate of the upper left pixel of the cropbox
            ex : int
                the x-coordinate of the lower right pixel of the cropbox
            ey : int
                the y-coordinate of the lower right pixel of the cropbox

        Return
        ------
        pair = (sub_window, cropbox)
        sub_window : PIL.Image
            The resized cropped image
        cropbox : the box itself
        """
        sub_window = image.crop(cropbox).resize((self._target_width,
                                                self._target_height),
                                                self._interpolation)
        return sub_window, cropbox

    def get_final_size(self):
        """
        Return the final size of the windows

        Return
        ------
        pair = (height, width)
            height : int > 0
                The height of the subwindows
            width : int > 0
                The width of the subwindows
        """
        return self._target_height, self._target_width



# class SubWindowExtractor:
#     """
#     ==================
#     SubWindowExtractor
#     ==================
#     A :class:`SubWindowExtractor` extract subwindows from an image and resize
#     them to a given shape.

#     The size and location of a given subwindow are drawn randomly

#     Constructor parameters
#     ----------------------
#     min_size : float 0 < min_size <= 1
#         The minimum size of subwindow express as the size ratio with
#         the original image
#     max_size : float min_size <= max_size <= 1
#         The maximum size of subwindow express as the size ratio with
#         the original image
#     target_width : int > 0
#         The width of the subwindow after resizing
#     target_height : int > 0
#         the height of the subwindow after resizing
#     interpolation : int in {0, 1, 2, 3, 4, 5} (default: 1)
#         The order of the spline interpolation
#     number_generator : :class:`NumberGenerator`
#         The random number generator used for drawing the subwindows. It
#         draws the height and width of the subwindow (respecting the
#         original ratio) and then draws the location. Real number generator
#         are fine and will be casted into int
#     """


#     def __init__(self, min_size, max_size, target_width, target_height,
#                  interpolation, number_generator):

#         self._min_size = min_size
#         self._max_size = max_size
#         self._target_width = target_width
#         self._target_height = target_height
#         self._numGen = number_generator
#         self._interpolation = interpolation


#     def get_crop_box(self, width, height):
#         """
#         Draws a new crop box

#         Paramters
#         ---------
#         width : int > 0
#             the width of the image on which the cropbox will be used
#         height: int > 0
#             the height of the image on which the cropbox will be used

#         Return
#         ------
#         tuple = (sx, sy, ex, ey)
#         sx : int
#             The x-coordinate of the upper left pixel of the cropbox
#             (included)
#         sy : int
#             The y-coordinate of the upper left pixel of the cropbox
#             (included)
#         ex : int
#             the x-coordinate of the lower right pixel of the cropbox
#             (excluded)
#         ey : int
#             the y-coordinate of the lower right pixel of the cropbox
#             (excluded)
#         """
#         if width < height:
#             ratio = 1. * self._target_height / self._target_width
#             min_width = self._min_size * width
#             max_width = self._max_size * width

#             if min_width * ratio > height:
#                 raise ValueError

#             if max_width * ratio > height:
#                 max_width = height / ratio

#             crop_width = self._numGen.get_number(min_width, max_width)
#             crop_height = ratio * crop_width

#         else:
#             ratio = 1. * self._target_width / self._target_height
#             min_height = self._min_size * height
#             max_height = self._max_size * height

#             if min_height * ratio > width:
#                 raise ValueError

#             if max_height * ratio > width:
#                 max_height = width / ratio

#             crop_height = self._numGen.get_number(min_height, max_height)
#             crop_width = ratio * crop_height

#         if crop_width == 0:
#             crop_width = 1
#         if crop_height == 0:
#             crop_height = 1

#         # Draw a random position
#         sx = int(self._numGen.get_number(0, width-crop_width))
#         sy = int(self._numGen.get_number(0, height-crop_height))

#         # Crop subwindow
#         return (sx, sy, int(sx+crop_width), int(sy+crop_height))

#     def extract_with_boxes(self, image):
#         """
#         Extract a subwindow of an image

#         Parameters
#         ----------
#         image : skimage
#             The image from which to extract the subwindow

#         Return
#         ------
#         pair = (subwindow, box)
#         subwindow : skimage
#             The subwindow extracted from the original image
#         box = (sx, sy, ex, ey) the croping box
#             sx : int
#                 The x-coordinate of the upper left pixel of the cropbox
#                 (included)
#             sy : int
#                 The y-coordinate of the upper left pixel of the cropbox
#                 (included)
#             ex : int
#                 the x-coordinate of the lower right pixel of the cropbox
#                 (excluded)
#             ey : int
#                 the y-coordinate of the lower right pixel of the cropbox
#                 (excluded)
#         """
#         # Draw a random window
#         height = image.shape[0]
#         width = image.shape[1]
#         try:
#             box = self.get_crop_box(width, height)
#         except CorpLargerError:
#             #subwindow larger than image, so we simply resize original image
#             #to target sizes
#             sub_window = resize(image,
#                                 (self._target_width, self._target_height),
#                                 self._interpolation)
#             return sub_window, box

#         return self.crop_and_resize(image, box)

#     def extract(self, image):
#         """
#         Extract a subwindow of an image

#         Parameters
#         ----------
#         image : PIL.Image
#             The image from which to extract the subwindow

#         Return
#         ------
#         subwindow : PIL.Image
#             The subwindow extracted from the original image
#         """
#         sw, box = self.extract_with_boxes(image)
#         return sw

#     def crop_and_resize(self, image, cropbox):
#         """
#         Apply image cropping and resize image thanks to the instance
#         reinterpolation mechanism

#         Parameters
#         ----------
#         image : PIL.Image
#             The image from which to extract the subwindow
#         cropbox = (sx, sy, ex, ey) a *valid* croping box :
#             sx : int
#                 The x-coordinate of the upper left pixel of the cropbox
#             sy : int
#                 The y-coordinate of the upper left pixel of the cropbox
#             ex : int
#                 the x-coordinate of the lower right pixel of the cropbox
#             ey : int
#                 the y-coordinate of the lower right pixel of the cropbox

#         Return
#         ------
#         pair = (sub_window, cropbox)
#         sub_window : PIL.Image
#             The resized cropped image
#         cropbox : the box itself
#         """
#         sx, sy, ex, ey = cropbox

#         sub_window = resize(image[sx:ex, sy:ey],
#                             (self._target_width, self._target_height),
#                             self._interpolation)
#         return sub_window, cropbox

#     def get_final_size(self):
#         """
#         Return the final size of the windows

#         Return
#         ------
#         pair = (height, width)
#             height : int > 0
#                 The height of the subwindows
#             width : int > 0
#                 The width of the subwindows
#         """
#         return self._target_height, self._target_width


class FixTargetSWExtractor(SubWindowExtractor):
    """
    ====================
    FixTargetSWExtractor
    ====================
    This subwindow extractor does not draw the size of the subwindow but
    directly uses the target size.
    """
    def __init__(self, target_width, target_height, interpolation,
                 number_generator):
        """
        Construct a :class:`FixTargetSWExtractor` instance.

        Parameters
        ----------
        target_width : int > 0
            The width of the subwindow after resizing
        target_height : int > 0
            the height of the subwindow after resizing
        interpolation : int {INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR,
        INTERPOLATION_CUBIC, INTERPOLATION_ANTIALIAS}
            The reintorpolation mechanism
        number_generator : :class:`NumberGenerator`
            The random number generator used for drawing the subwindow
            locations. Real number generator are fine and will be casted
            into int
        """
        self._target_height = target_height
        self._target_width = target_width
        self._numGen = number_generator
        self.set_interpolation(interpolation)

    def get_crop_box(self, width, height):

        crop_width = self._target_width
        crop_height = self._target_height
        if crop_width > width or crop_height > height:
            raise CorpLargerError("Crop larger than image")

         # Draw a random position
        sx = int(self._numGen.get_number(0, width-crop_width))
        sy = int(self._numGen.get_number(0, height-crop_height))

        # Crop subwindow
        return (sx, sy, int(sx + crop_width), int(sy + crop_height))


class FixImgSWExtractor(SubWindowExtractor):
    """
    ====================
    FixImgSWExtractor
    ====================
    This subwindow extractor works with images of a given width and height
    """
    def __init__(self, image_width, image_height, min_size, max_size,
                 target_width, target_height, interpolation,
                 number_generator):
        """
        Construct a :class:`FixImgSWExtractor`

        Parameters
        ----------
        image_width : int > 0
            The image width
        image_height : int > 0
            the image height
        min_size : float 0 < min_size <= 1
            The minimum size of subwindow express as the size ratio with
            the original image
        max_size : float min_size <= max_size <= 1
            The maximum size of subwindow express as the size ratio with
            the original image
        target_width : int > 0
            The width of the subwindow after resizing
        target_height : int > 0
            the height of the subwindow after resizing
        interpolation : int {INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR,
        INTERPOLATION_CUBIC, INTERPOLATION_ANTIALIAS}
            The reintorpolation mechanism
        number_generator : :class:`NumberGenerator`
            The random number generator used for drawing the subwindows. It
            draws the height and width of the subwindow (respecting the
            original ratio) and then draws the location. Real number generator
            are fine and will be casted into int
        """
        self._imgWidth = image_width
        self._imgHeight = image_height
        self._min_size = min_size
        self._max_size = max_size
        self._target_width = target_width
        self._target_height = target_height
        self._numGen = number_generator
        self.set_interpolation(interpolation)
        self._computeMinMaxHeight(image_width, image_height)

    def _computeMinMaxHeight(self, width, height):
        ratio = 1. * self._target_width / self._target_height
        min_height = self._min_size * height
        max_height = self._max_size * height

        if min_height * ratio > width:
            raise ValueError

        if max_height * ratio > width:
            max_height = width / ratio

        self._minHeight = min_height
        self._maxHeight = max_height
        self._ratio = ratio

    def get_crop_box(self, width=None, height=None):

        crop_height = self._numGen.get_number(self._minHeight, self._maxHeight)
        crop_width = self._ratio * crop_height

        if crop_width == 0:
            crop_width = 1
        if crop_height == 0:
            crop_height = 1

        # Draw a random position
        sx = int(self._numGen.get_number(0, self._imgWidth-crop_width))
        sy = int(self._numGen.get_number(0, self._imgHeight-crop_height))

        # Crop subwindow
        return (sx, sy, int(sx + crop_width), int(sy + crop_height))


class CorpLargerError(Exception):
    """
    ===============
    CorpLargerError
    ===============
    An exception class which represents the fact that a cropping box is
    larger than the image to crop
    """
    def __init__(self, value):
        Exception.__init__(self, value)

    def __str__(self):
        return repr(self.value)


class MultiSWExtractor:
    """
    ================
    MultiSWExtractor
    ================
    A subwindow extractor which extracts severals subwindows per image.

    See :meth:`refresh`.
    """
    def __init__(self, subwindowExtractor, nb_subwindows, autoRefresh=False):
        """
        Construct a :class:`MultiSWExtractor`

        Parameters
        ----------
        subwindowExtractor : :class:`SubWindowExtractor`
            The instance which will extract each subwindow
        nb_subwindows : int > 0
            The number of subwindow to extract
        autoRefresh : boolean (default : False)
            if true, refreshes the set of cropboxes for each image
            See :meth:`refresh`.
        """
        self._sw_extractor = subwindowExtractor
        self._nb_sw = nb_subwindows
        self._autoRefresh = autoRefresh

    def __len__(self):
        return self._nb_sw

    def nb_subwidows(self):
        """
        Return the number of subwindows that this instance will extract
        """
        return self._nb_sw

    def refresh(self, width, height):
        """
        Change/refresh/update the set of cropboxes.

        Parameters
        ----------
        width : int > 0
            The image width
        height : int > 0
            the image height
        """
        boxes = []
        for i in xrange(self._nb_sw):
            boxes.append(self._sw_extractor.get_crop_box(width, height))
        self._boxes = boxes

    def extract_with_boxes(self, image):
        """
        Extract a subwindow of an image

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindows

        Return
        ------
        list : list of pairs = (subwindow, box)
        subwindow : PIL.Image
            The subwindow extracted from the original image
        box = (sx, sy, ex, ey) the croping box
        sx : int
            The x-coordinate of the upper left pixel of the cropbox
        sy : int
            The y-coordinate of the upper left pixel of the cropbox
        ex : int
            the x-coordinate of the lower right pixel of the cropbox
        ey : int
            the y-coordinate of the lower right pixel of the cropbox
        """
        #Testing auto refresh
        if self._autoRefresh:
            width, height = image.size
            self.refresh(width, height)
        #Extracting the boxes
        subwindowsAndBoxes = []
        for box in self._boxes:
            subwindowsAndBoxes.append(
                self._sw_extractor.crop_and_resize(image, box))
        return subwindowsAndBoxes

    def extract(self, image):
        """
        Extract a subwindow of an image

        Parameters
        ----------
        image : PIL.Image
            The image from which to extract the subwindows

        Return
        ------
        list of subwindows
        subwindow : PIL.Image
            The subwindow extracted from the original image
        """
        #Testing auto refresh
        if self._autoRefresh:
            width, height = image.size
            self.refresh(width, height)
        #Extracting the boxes
        sub_windows = []
        for box in self._boxes:
            sw, _ = self._sw_extractor.crop_and_resize(image, box)
            sub_windows.append(sw)
        return sub_windows

    def get_final_size(self):
        """
        Return the final size of the windows

        Return
        ------
        pair = (height, width)
            height : int > 0
                The height of the subwindows
            width : int > 0
                The width of the subwindows
        """
        return self._sw_extractor.get_final_size()


if __name__ == "__main__":

    test = True

    if test:
        imgpath = "lena.png"
        from NumberGenerator import NumberGenerator
        try:
            import Image
        except:
            from PIL import Image
        img = Image.open(imgpath)
        width, height = img.size
        swExt = SubWindowExtractor(0.5,1.,256,250, SubWindowExtractor.INTERPOLATION_BILINEAR, NumberGenerator())

        sw1,box1 = swExt.extract(img)

        mExt = MultiSWExtractor(swExt, 10)
        mExt.refresh(width, height)

        sws = mExt.extract(img)



        swExt = FixImgSWExtractor(width, height, 0.5,1.,256,250, SubWindowExtractor.INTERPOLATION_BILINEAR, NumberGenerator())

        sw2,box2 = swExt.extract(img)

        mExt = MultiSWExtractor(swExt, 10)
        mExt.refresh(width, height)

        sws2 = mExt.extract(img)
