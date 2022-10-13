from abc import abstractmethod

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Dict

from wsipipe.load.annotations import AnnotationSet
from wsipipe.load.slides import SlideBase
 

class Loader(metaclass=ABCMeta):
    """Generic Loader class

    Returns:
        name (str): Name of the loader.
        load_annotations (object): A function used to load annotations for the dataset
        load_slide (object): A function used to load slides for the dataset
        labels (Dict[str: int]): A dictionary of category names and the corresponding integer label for the dataset
    """
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_annotations(self, file: Path) -> AnnotationSet:
        raise NotImplementedError

    @abstractmethod
    def load_slide(self, path: Path) -> SlideBase:
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "normal": 1, "tumor": 2}
