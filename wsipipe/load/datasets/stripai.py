"""
Loader for the STRIP AI dataset.
    - Slides are tiffs read using openslide
    - A single annotation is applied to whole slide
    - Output labels for slides are background, CE and LAA

"""

from pathlib import Path
from typing import Dict

from wsipipe.load.annotations import AnnotationSet
from wsipipe.load.datasets.loader import Loader
from wsipipe.load.slides import OSSlide, SlideBase  


class StripaiLoader(Loader):
    @property
    def name(self) -> str:
        return "StripaiLoader"

    def load_annotations(self, label: Path) -> AnnotationSet:
        # if there is no annotation file the just pass and empty list
        annotations = []
        labels_order = ["background", "Other", "CE", "LAA"]
        label = str(label.stem)
        return AnnotationSet(annotations, self.labels, labels_order, label)

    def load_slide(self, path: Path) -> SlideBase:
        return OSSlide(path)

    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "CE": 1, "LAA": 2, "Other": 3}