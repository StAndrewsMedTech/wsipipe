"""
Transforms can be applied to binary or labelled images, for example to fill holes 
"""

from abc import ABCMeta, abstractmethod
import math

import numpy as np
from skimage.measure import label, regionprops, block_reduce
from skimage.morphology import binary_closing, binary_opening


class MorphologyTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.array:
        raise NotImplementedError

class NullTransform(MorphologyTransform):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Does not apply a transform """
        return image

class SimpleClosingTransform(MorphologyTransform):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Applies binary closing transform (fills small holes in a binary image)"""
        mask_out = binary_closing(image)
        return mask_out

class SimpleOpeningTransform(MorphologyTransform):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Applies binary opening transform (removes small bright dots in a binary image)"""
        mask_out = binary_opening(image)
        return mask_out

class SizedClosingTransform(MorphologyTransform):
    def __init__(self, level_in: int, expand_size: float = 50, level_zero_size: float = 0.25) -> None:
        # assign values
        self.level_in = level_in
        self.expand_size = expand_size
        self.level_zero_size = level_zero_size
        
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """ Applies binary closing transform with a neighbourhood of specified size (fills holes smaller than specified size)"""
        area_of_pixel = 2 ** self.level_in * self.level_zero_size
        pixels_to_expand = math.ceil(self.expand_size / area_of_pixel)
        neighbourhood = np.ones((pixels_to_expand, pixels_to_expand))
        mask_out = binary_closing(image, neighbourhood)
        return mask_out

class FillHolesTransform(MorphologyTransform):
    """Fills holes in an image, using segmentation
    Segments smaller than hole_size_to_fill in area are filled.  
    Size of a pixel at the image level is 2**level_in * level zero size
    Hole_size_to_fill (an area) is converted to number of pixels by
    dividing by the size of rpixel at image level.

    Input image is a binary image 
    Image is segmented using scikit image regionprops.
    If the area of the region is less than the specified hole size 
    and the mean intensity of the region is less than 0.1 (out of 1)
    then the region is filled by converting to True/1/white 

    Args:
    level_in: level of input image 
    hole_size_to_fill: dark areas smaller in size than this will be filled
    level_zero_size: size of a pixel at level zero
    
    """
    def __init__(self, level_in: int, hole_size_to_fill: float = 250, level_zero_size: float = 0.25) -> None:
        # assign values
        self.level_in = level_in
        self.hole_size_to_fill = hole_size_to_fill
        self.level_zero_size = level_zero_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        area_of_pixel = 2 ** self.level_in * self.level_zero_size
        pixel_size_to_fill = self.hole_size_to_fill / area_of_pixel

        def fill_region_or_not(reg):
            size_to_fill = reg.area < pixel_size_to_fill
            colour_to_fill = reg.mean_intensity < 0.1
            fill_reg = size_to_fill & colour_to_fill
            return fill_reg

        # set background bigger than pixel value so for this nothing is counted as background
        label_image = label(image, background=256)
        regions = regionprops(label_image, image)
        regions_to_fill_mask = [fill_region_or_not(reg) for reg in regions]
        regions_to_fill = np.add(np.arange(len(regions)),1)[regions_to_fill_mask]

        mask_to_fill = np.isin(label_image, regions_to_fill)
        filled_holes = np.where(mask_to_fill, True, image)

        return filled_holes

class MaxPoolTransform(MorphologyTransform):
    """ Applies max pool
    Takes a big input image and returns a smaller output image. 
    Every pixel in the output image represents 2**(level_out - level_in) pixels in input image.
    The pixel value for the output image is the maximum of the pixels in that region of the input image.

    Args: 
    level_in: Initial level of image 
    level_out: Output level of image (must be a smaller image level_out > level_in) 
    """
    def __init__(self, level_in: int, level_out: int) -> None:
        # assign values
        self.level_in = level_in
        self.level_out = level_out
        
    def __call__(self, image: np.ndarray) -> np.ndarray:

        pool_size = 2 ** (self.level_out - self.level_in)
        image_out = block_reduce(image, (pool_size, pool_size), func=np.max)
        return image_out