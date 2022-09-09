"""
Filters to apply to images as part of tissue detection
"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


class PreFilter(metaclass=ABCMeta):
    """ Generic class of filter """
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError

class NullBlur(PreFilter):
    """ Null filter does nothing """
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image

class MedianBlur(PreFilter):
    """ Applies a median filter of size filter_size """
    def __init__(self, filter_size: int) -> None:
        # assign values
        self.filter_size = filter_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image_out = median_filter(image, size=self.filter_size)
        return image_out

class GaussianBlur(PreFilter):
    """ Applies a Gaussian filter with sigma value """
    def __init__(self, sigma: int) -> None:
        # assign values
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image_out = gaussian_filter(image, sigma=self.sigma) 
        return image_out      