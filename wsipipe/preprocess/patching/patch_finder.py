"""
Patch Finders describe how patches are created for a slide.

They work on a labelled image, that is a numpy array with
integers giving the annotation category for each pixel.

The input labelled image can be at any level of the pyramid 
for which a numpy array for that size can fit into memory.

A patch finder will create a dataframe with columns x, y, label
where x and y represents the top left corner of the patch and
label is the label applied to the patch.

"""

from abc import ABCMeta, abstractmethod
from math import ceil
from random import randint
from typing import Tuple

import numpy as np
import pandas as pd

from wsipipe.utils import to_frame_with_locations, pool2d, Size


class PatchFinder(metaclass=ABCMeta):
    """Generic patch finder class

    Args:
        labels_image (np.array): The whole slide image represented as a 2d numpy array, 
            the classification is given by an integer. For example an image such as those
            output by AnnotationSet.render
        slide_shape (Size): The size of the WSI at the level at which the labels are rendered. 
            This may be different to the labels image shape, as the labels image may not
            include blank parts of the slide in the bottom right.
    """
    @abstractmethod
    def __call__(
        self, labels_image: np.array, slide_shape: Size
    ) -> Tuple[pd.DataFrame, int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def labels_level(self):
        raise NotImplementedError


class GridPatchFinder(PatchFinder):
    def __init__(
        self,
        labels_level: int,
        patch_level: int,
        patch_size: int,
        stride: int,  # defined in terms of the labels image space
        border: int = 0,
        jitter: int = 0,
        remove_background: bool = True,
        pool_mode: str = "max"
    ) -> None:
        """ Grid Patch Finder Class

        This creates patches on a regular grid.
        
        The assumption is that the same settings will be used for a number of different slides.
        The class returns an object that can then be called on multiple slides.
        It is assumed background of the slide is labelled 0, and all other categories are greater than 0

        Labels level specifies the magnification level at which the labels image is create. 
        Patch level is the level at which patches are to be created. These values along with 
        the patch size are used to calculate how many pixels in the labels image are in a patch.
        A pooling operation is then applied that reduces the labelled image based on the patch size and stride. 
        Each pixel in the pooled image is then equivalent to a patch and a dataframe is constructed of 
        row column and label for every pixel in pooled image. 
        Rows and columns are converted to x and y values at the patch level, which gives the top left
        position of the patches. Adjustments are then applied to account for borders and jitter of
        patches. A check is applied to make sure no patches have been defined at the edges that will fall outside of
        the overall slide size, if so they are shifted inwards so they align with the edge of the slide.
        Finally if the background patches are not required they are removed from the dataframe.

        Args:
            labels_level (int): The magnification level of the labels image.
            patch_level (int): The magnification level at which to extract the pixels data for the patches.
            patch_size (int): The width and height of the patches in pixels at patches_level magnification.
            stride (int): The horizontal and vertical distance between each patch (the stride of the window).
            border (int, optional): This is a border added on to the patch_size (half on each side). 
                Therefore the overall patch label and stride is based on the central patch size area, 
                but the total patch size is larger. Defaults to 0.
            jitter (int, optional): Adds jitter to the overall patch size so that a random jitter can be applied 
                within the patch to give the final patch size. For example a patch size of 256, with a jitter of 8
                would output patches of 264. So that every epoch of training a random jitter of up to 8 pixels 
                could be applied and still output 256 patches from the 264. Defaults to 0.
            remove_background (bool, optional): Switch to keep the background patches in dataframe or not. Defaults to True
            pool_mode (str, optional): Specifies how multiple pixels per patch are to be pooled to give an overall label.
                Values are "max", "mode". Mode gives the most common label in the patch. 
                Max assumes that higher labels are more important and therefore returns the highest label.
                Defaults to "max"

        """

        # assign values
        self.labels_level = labels_level
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.stride = stride
        self.border = border
        self.jitter = jitter
        self.remove_background = remove_background
        self.pool_mode = pool_mode
        # some assumptions
        # 1. patch_size is some integer multiple of a pixel at labels_level
        # 2. patch_level is equal to or below labels_level
        # 3. stride is some integer multiple of a pixel at labels_level

    def __call__(
        self, labels_image: np.array, slide_shape: Size
    ) -> Tuple[pd.DataFrame, int, int]:
        """Grid patch finder call for individual slide.

        Args:
            labels_image (np.array): An array containing a label index for each pixel at some magnification level.
            slide_shape (Size): The overall size of the slide at level of patch extraction, 
                used to check patches aren't created outside of this size.
        Returns:
            df (pd.Dataframe): Dataframe with x, y, label containing top left position of patch plus label
            patch_level (int): magnification level at which aptches were created
            output_patch_size (int): final size at which patches were created
        """
        scale_factor = 2 ** (self.labels_level - self.patch_level)
        kernel_size = int(self.patch_size / scale_factor)
        label_level_stride = int(self.stride / scale_factor)

        # TODO - Needs to select no the max label but the one with the most area? - needs thinking about this!
        # The pooling operation might be a parameter for the patch finder.
        patch_labels = pool2d(labels_image, kernel_size, label_level_stride, 0, pool_mode=self.pool_mode)

        # convert the 2d array of patch labels to a data frame
        df = to_frame_with_locations(patch_labels, "label")
        df.row *= self.patch_size
        df.column *= self.patch_size
        df = df.rename(columns={"row": "y", "column": "x"})
        df = df.reindex(columns=["x", "y", "label"])

        # calculate amount to subtract from top left for border and jitter
        subtract_top_left = ceil(self.border / 2) + self.jitter

        # for each row, add the border
        df["x"] = np.subtract(df["x"], subtract_top_left)
        df["y"] = np.subtract(df["y"], subtract_top_left)
        output_patch_size = self.patch_size + (self.border + self.jitter)

        # remove the background
        if self.remove_background:
            df = df[
                df.label != 0
            ]  # TODO: put this in as a method that is optional on the slide patch index (or something)

        # clip the patch coordinates to the slide dimensions
        df["x"] = np.maximum(df["x"], 0)
        df["y"] = np.maximum(df["y"], 0)
        df["x"] = np.minimum(df["x"], slide_shape.width - output_patch_size)
        df["y"] = np.minimum(df["y"], slide_shape.height - output_patch_size)

        # return the index and the data required to extract the patches later
        return df, self.patch_level, output_patch_size

    def labels_level(self):
        raise self.labels_level


class RandomPatchFinder(PatchFinder):
    def __init__(
        self,
        labels_level: int,
        patch_level: int,
        patch_size: int,
        border: int = 0,
        npatches: int = 1000,
        pool_mode: str = "mode"
    ) -> None:
        """ Random Patch Finder Class

        This creates the specifed number of patches at random points on the slide
        
        The assumption is that the same settings will be used for a number of different slides.
        The class returns an object that can then be called on multiple slides.
        It is assumed background of the slide is labelled 0, and all other categories are greater than 0

        Labels level specifies the magnification level at which the labels image is create. 
        Patch level is the level at which patches are to be created. 
        A random x, y position within the slide size is generated at patch level. 
        The x y position is converted to the labels level. 
        All the pixels in the labels within the patch are pooled to determine an overall label.
        If the overall patch label is greater than zero (not background) then it is written to the dataframe, 
        else it is rejected. This is repeated until npatches that are not background are found.        
        The x and y values at the patch level, gives the top left position of the patches. 
        Adjustments are then applied to account for borders and jitter of
        patches. A check is applied to make sure no patches have been defined at the edges that will fall outside of
        the overall slide size, if so they are shifted inwards so they align with the edge of the slide.


        Args:
            labels_level (int): The magnification level of the labels image.
            patch_level (int): The magnification level at which to extract the pixels data for the patches.
            patch_size (int): The width and height of the patches in pixels at patches_level magnification.
            border (int, optional): This is a border added on to the patch_size (half on each side). 
                Therefore the overall patch label and stride is based on the central patch size area, 
                but the total patch size is larger. Defaults to 0.
            npatches (int, optional): The number of patches to create per slide. Defaults to 1000.
            pool_mode (str, optional): Specifies how multiple pixels per patch are to be pooled to give an overall label.
                Values are "max", "mode". Mode gives the most common label in the patch. 
                Max assumes that higher labels are more important and therefore returns the highest label.
                Defaults to "mode"

        """

        # assign values
        self.labels_level = labels_level
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.border = border
        self.npatches = npatches
        self.pool_mode = pool_mode

    def __call__(
        self, 
        labels_image: np.array, 
        slide_shape: Size
    ) -> Tuple[pd.DataFrame, int, int]:
        """Random patch finder call for individual slide.

        Args:
            labels_image (np.array): An array containing a label index for each pixel at some magnification level.
            slide_shape (Size): The overall size of the slide at level of patch extraction, 
                used to check patches aren't created outside of this size.
        Returns:
            df (pd.Dataframe): Dataframe with x, y, label containing top left position of patch plus label
            patch_level (int): magnification level at which aptches were created
            output_patch_size (int): final size at which patches were created
        """
        def get_mode(arr: np.array) -> int:
            labels = np.unique(arr)
            counts = [np.sum(arr==lab) for lab in labels]
            mode = labels[np.argmax(counts)]
            return mode
        
        patchcount = 0
        patches = np.zeros((self.npatches, 3))
        levdiff = self.labels_level - self.patch_level
        pixeldiff = 2 ** levdiff
        pixelsperpatch = self.patch_size // pixeldiff
        while patchcount < self.npatches:
            rowx = randint(0, slide_shape.width)
            rowy = randint(0, slide_shape.height)
            testx = int(rowx // pixeldiff) 
            testy = int(rowy // pixeldiff)
            testxmx = testx + pixelsperpatch
            testymx = testy + pixelsperpatch
            if testx >= labels_image.shape[1]:
                continue
            if testy >= labels_image.shape[0]:
                continue
            testlabels = labels_image[testy:testymx, testx:testxmx]
            if self.pool_mode == "mode":
                testlabel = get_mode(testlabels)
            else:
                testlabel = np.max(testlabels)
            if testlabel > 0:
                patches[patchcount, :] = [rowx, rowy, testlabel]
                patchcount += 1
                
        df = pd.DataFrame(patches, columns=["x", "y", "label"]).astype(int)

        # calculate amount to subtract from top left for border
        subtract_top_left = ceil(self.border / 2)

        # for each row, add the border
        df["x"] = np.subtract(df["x"], subtract_top_left)
        df["y"] = np.subtract(df["y"], subtract_top_left)
        output_patch_size = self.patch_size + self.border

        # clip the patch coordinates to the slide dimensions
        df["x"] = np.maximum(df["x"], 0)
        df["y"] = np.maximum(df["y"], 0)
        df["x"] = np.minimum(df["x"], slide_shape.width - output_patch_size)
        df["y"] = np.minimum(df["y"], slide_shape.height - output_patch_size)

        # return the index and the data required to extract the patches later
        return df, self.patch_level, output_patch_size

    def labels_level(self):
        raise self.labels_level
