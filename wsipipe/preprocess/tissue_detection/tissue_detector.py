"""
Tissue Detectors create a 2d array of booleans indicating if that area contains
tissue or not. 

The input is an RGB numpy array representing the slide.
Usually a downsampled thumbnail image as whole slide images as level 0
are often too large to store in memory.

"""

from abc import ABCMeta, abstractmethod
from typing import List, Union

import numpy as np
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters import threshold_otsu

from wsipipe.preprocess.tissue_detection.filters import PreFilter, NullBlur
from wsipipe.preprocess.tissue_detection.morphology_transforms import MorphologyTransform, NullTransform

class TissueDetector(metaclass=ABCMeta):
    """Generic tissue detector class

    Args:
        pre_filter: Any filters or transforms that are to be applied before the tissue detection.
            Can be lists of filters or individual filters. Defaults to NullBlur
        morph_transform (): Any filters or transforms that are to be applied after the tissue detection.
            Can be lists of transforms or individual transforms. Defaults to NullTransform
    Returns:
        An ndarray of booleans with the same dimensions as the input image
        True means foreground, False means background
    """
    def __init__(self, 
        pre_filter: Union[PreFilter, List[PreFilter]] = NullBlur(), 
        morph_transform: Union[MorphologyTransform, List[MorphologyTransform]] = NullTransform()
    ) -> None:
        # assign values
        if not isinstance(pre_filter, list):
            self.pre_filter = [pre_filter]
        else:
            self.pre_filter = pre_filter
        if not isinstance(morph_transform, list):
            self.morph_transform = [morph_transform]
        else:
            self.morph_transform = morph_transform

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError


class TissueDetectorOTSU(TissueDetector):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Applies Otsu thresholding as a tissue detector

        1. Convert from RGB to HSV
        2. Perform automatic thresholding using Otsu's method on the H and S channels
        3. Combine the thresholded H and S channels

        Also applies prefilters and post transforms if specified

        Args:
            image: A scaled down WSI image. Must be r,g,b.
        Returns:
            An ndarray of booleans with the same dimensions as the input image
            True means foreground, False means background
        """
        # convert the image into the hsv colour space
        image_hsv = rgb2hsv(image)

        # apply filter
        for pf in self.pre_filter:
            image_hsv = pf(image_hsv)

        # use Otsu's method to find the thresholds for hue and saturation
        thresh_h = threshold_otsu(image_hsv[:, :, 0])
        thresh_s = threshold_otsu(image_hsv[:, :, 1])

        # mask the image to get determine which pixels with hue and saturation above their thresholds
        mask_h = image_hsv[:, :, 1] > thresh_h
        mask_s = image_hsv[:, :, 1] > thresh_s

        # combine the masks with an OR so any pixel above either threshold counts as foreground
        np_mask = np.logical_or(mask_h, mask_s)

        # apply morphological transforms
        for mt in self.morph_transform:
            np_mask = mt(np_mask)

        return np_mask


class TissueDetectorGreyScale(TissueDetector):
    def __init__(self, pre_filter = NullBlur(), morph_transform = NullTransform(), grey_level: float = 0.8) -> None:
        """ Greyscale tissue detector 
        
        In addition to prefilter and morph_transform from generic class

        Args: 
            grey_level (float): Value between 0 and 1 for greyscale threshold. Defaults to 0.8
        """
        super().__init__(pre_filter, morph_transform)
        self.grey_level = grey_level

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Applies greyscale thresholding as a tissue detector

        1. Convert PIL image to numpy array
        2. Convert from RGB to gray scale
        3. Get masks, any pixel that is less than a threshold (e.g. 0.8)

        Also applies prefilters and post transforms if specified

        Args:
        image: A scaled down WSI image. Must be r,g,b.
        Returns:
        An ndarray of booleans with the same dimensions as the input image
        True means foreground, False means background
        """
        # convert PIL image to numpy array
        image = np.asarray(image)

        # change pure black to pure white
        imager = image[:, :, 0] == 0
        imageg = image[:, :, 1] == 0
        imageb = image[:, :, 2] == 0
        image_mask = np.expand_dims(np.logical_and(np.logical_and(imager, imageg), imageb), axis=-1)
        image = np.where(image_mask, [255,255,255], image)
        image = np.array(image, dtype=np.uint8)

        # convert to gray-scale
        image_grey = rgb2gray(image)

        # apply filter
        for pf in self.pre_filter:
            image = pf(image)

        # get masks, any pixel that is less than 0.8
        np_mask = np.less_equal(image_grey, self.grey_level)

        # apply morphological transforms
        for mt in self.morph_transform:
            np_mask = mt(np_mask)

        return np_mask


class TissueDetectorAll(TissueDetector):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """A Null tissue detector

        Just returns the whole image as True everything is foreground to be kept

        Args:
        image: A scaled down WSI image. Must be r,g,b.
        Returns:
        An ndarray of booleans with the same dimensions as the input image
        True means foreground
        """
        # convert PIL image to numpy array
        image = np.asarray(image)

        # get masks, all pixels
        np_mask = np.array(np.ones(image.shape[0:2]), dtype=bool)

        return np_mask