"""
Loader for the Camelyon 16 dataset.
    - Slides are tiffs read using openslide
    - Annotations are asapxml
    - Output labels for slides are background, normal and tumor

"""

from pathlib import Path
from typing import Dict

from wsipipe.load.annotations import AnnotationSet, load_annotations_asapxml
from wsipipe.load.datasets.loader import Loader
from wsipipe.load.slides import OSSlide, SlideBase  


class Camelyon16Loader(Loader):
    @property
    def name(self) -> str:
        return "Camelyon16Loader"

    def load_annotations(self, file: Path) -> AnnotationSet:
        # if there is no annotation file the just pass and empty list
        group_labels = {
            "Tumor": "tumor",
            "_0": "tumor",
            "_1": "tumor",
            "_2": "normal",
            "Exclusion": "normal",
            "None": "normal",
        }
        annotations = load_annotations_asapxml(file, group_labels) if file else []
        labels_order = ["background", "tumor", "normal"]
        return AnnotationSet(annotations, self.labels, labels_order, "normal")

    def load_slide(self, path: Path) -> SlideBase:
        return OSSlide(path)

    @property
    def labels(self) -> Dict[str, int]:
        return {"background": 0, "normal": 1, "tumor": 2}