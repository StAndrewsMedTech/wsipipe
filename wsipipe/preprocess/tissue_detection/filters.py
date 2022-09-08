from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


class PreFilter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError

class NullBlur(PreFilter):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image

class MedianBlur(PreFilter):
    def __init__(self, filter_size: int) -> None:
        # assign values
        self.filter_size = filter_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image_out = median_filter(image, size=self.filter_size)
        return image_out

class GaussianBlur(PreFilter):
    def __init__(self, sigma: int) -> None:
        # assign values
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image_out = gaussian_filter(image, sigma=self.sigma) 
        return image_out      