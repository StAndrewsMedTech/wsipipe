"""
Parent classes that contain functionality for reading annotations. 
These are used to render different types of annotations into a common format

"""

from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

from wsipipe.utils import PointF, Shape

annotation_types = ["Dot", "Polygon", "Spline", "Rectangle"]


class Annotation:
    """Class for a single annotation.

    There can be multiple annotations on a slide

    Args:
        name (str): Name of the annotation.
        type (str): One of Dot, Polygon, Spline or Rectangle
        label (str): What label should be given to the annotation
        vertices (List[PointF]): A list of vertices, each of which is an PointF object, 
            a named tuple (x, y) of floats.
    """
    def __init__(
        self, name: str, annotation_type: str, label: str, vertices: List[PointF]
    ):
        assert annotation_type in annotation_types
        self.name = name
        self.type = annotation_type
        self.label = label
        self.coordinates = vertices

    def draw(self, image: np.array, labels: Dict[str, int], factor: float):
        """Renders the annotation into the image.

        Args:
            image (np.array): Array to write the annotations into, must have dtype float.
            labels (Dict[str, int]): The value to write into the image for each type of label.
            factor (float): How much to scale (by divison) each vertex by.
        """
        fill_colour = labels[self.label]
        vertices = np.array(self.coordinates) / factor
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(image, [vertices], (fill_colour))


class AnnotationSet:
    """Class for all annotations on a slide.

    Args:
        annotations (List[Annotation]): A list of all Annotations on a slide
        labels (Dict[str, int]): A dictionary where the keys are the names of labels, 
            with the integer values with which the string should be replaced.
        labels_order (List[str]): An order the labels should be plotted in. 
            Where annotations overlap they will be drawn
            in this order, so the final label will be on top
        fill_label (str): The label given to any unannotated areas. 
    """
    def __init__(
        self,
        annotations: List[Annotation],
        labels: Dict[str, int],
        labels_order: List[str],
        fill_label: str,
    ) -> None:
        self.annotations = annotations
        self.labels = labels
        self.labels_order = labels_order
        self.fill_label = fill_label

    def render(self, shape: Shape, factor: float) -> np.array:
        """Creates a labelled image containing annotations

        This creates an array of size = shape, that is factor times smaller
        than the level at which the annotation vertexes are specified.
        Annotations vertex positions are assumed to be specified at level 0,
        and therefore for many WSI a np.array of the same size as level 0
        would not fit in memory. Therefore one factor times smaller is created.

        Args:
            shape (Shape): size of numpy array to create
            factor (float): How much to scale (by divison) each vertex by.
        """
        annotations = sorted(
            self.annotations, key=lambda a: self.labels_order.index(a.label)
        )
        image = np.full(shape, self.labels[self.fill_label], dtype=float)
        for a in annotations:
            a.draw(image, self.labels, factor)
        return image.astype("int")


def visualise_annotations(annot_path: Path, slide_path: Path, loader, level: int) -> np.array:
    """ Creates a image render of the annotations of a slide 
    
    Converts annotations from level zero to the specified level.
    Requires slide path to find the correct dimensions of the output image.
    Returns a numpy array

    Args:
        annot_path (Path): A path to the annotation file
        slide_path (Path): A path to the WSI file
        loader: The loader to use for slides and annots
        level (int): the level to create the numpy array
    Returns:
        labels_image (np.array): An array the same size as the WSI at level
        with the annotation labels plotted in it.
    """

    with loader.load_slide(slide_path) as slide:
        # read in annotations as list of x, y points at level zero
        annotations = loader.load_annotations(annot_path)
        # get shape of slide at level - size to plot the annotations
        labels_shape = slide.dimensions[level].as_shape()
        # get scale factor to convert annotations from level zero to level
        scale_factor = 2 ** level
        # render annotations into numpy array 
        labels_image = annotations.render(labels_shape, scale_factor)

    return labels_image