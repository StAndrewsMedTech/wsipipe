"""
PatchSets are sets of patches and all the information
required to create them from the slides.

Many patches in the set may use the same details, (which we call PatchSettings):
    - the path of the slide to read from
    - the level of the slide at which to create the patch
    - the size of the patch to be created
    - how to load the slide

To create an individual patch, you need to know:
    - the top left position of the patch
    - the label to be applied to the patch

Therefore the PatchSets are a dataframe and a settings list.

The settings list is a list of PatchSettings each of which contains: 
    slide_path, level, patch_size, loader
In the dataframe each row represents a patch and contains columns:
   x (top), y (left),  label, settings (index to list)

"""


from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
from typing import List

import pandas as pd
import cv2
import numpy as np

from wsipipe.load.datasets import Loader, get_loader
from wsipipe.load.slides import Region, SlideBase
from wsipipe.utils import invert

@dataclass
class PatchSetting:
    """Patch Setting Definition

    Args:
        level (int): The level at which patches are extracted
        patch_size (int): The size of patches to be created assumes square
        slide_path (Path): the path to the whole slide image
        loader (Loader): A method for loading the slide
    """
    level: int
    patch_size: int
    slide_path: Path  # not stored in the dataframe
    loader: Loader  # not stored in the dataframe

    def to_sdict(self):
        """Writes a PatchSetting to a dictionary so it can be saved to disk"""
        d = asdict(self)
        d["slide_path"] = str(self.slide_path)
        d["loader"] = self.loader.name
        return d

    @classmethod
    def from_sdict(cls, sdict: dict):
        """ Converts a dictionary to a PatchSetting"""
        sdict["slide_path"] = Path(sdict["slide_path"])
        sdict["loader"] = get_loader(sdict["loader"])
        return cls(**sdict)


class PatchSet:
    def __init__(self, df: pd.DataFrame, settings: List[PatchSetting]) -> None:
        """The dataframe should have the following columns:
            - x: left position of the patch at level
            - y: top position of the patch at level
            - label: which class it belongs to
            - setting: an index into the settings array.

        Args:
            df (pd.DataFrame): The patch locations, labels, and index into settings.
            settings (List[PatchSetting]): A list of settings.
        """
        self.df = df
        self.settings = settings

    def save(self, path: Path) -> None:
        """Saves a PatchSet to disk
        
        The dataframe is saved to a csv called frame.csv
        The settings are saved in a text file called settings.json

        Args:
            path (Path): the directory in which to save the patchset
        """
        path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path / "frame.csv", index=False)
        dicts = [s.to_sdict() for s in self.settings]
        with open(path / "settings.json", "w") as outfile:
            json.dump(dicts, outfile)

    @classmethod
    def load(cls, path: Path) -> "PatchSet":
        """Loads a PatchSet from disk
        
        Assumes:
        The dataframe is saved to a csv called frame.csv
        The settings are saved in a text file called settings.json

        Args:
            path (Path): the directory in which the patchset is saved
        """
        print(f"loading {path}")
        df = pd.read_csv(path / "frame.csv")
        with open(path / "settings.json") as json_file:
            settings = json.load(json_file)
            settings = [PatchSetting.from_sdict(s) for s in settings]
        return cls(df, settings)

    def export_patches(self, output_dir: Path) -> None:
        """Creates all patches in a patch set
        
        Writes patches in subdirectories of their label 
        Patches are name slide_path_x_y_level_patch_size.png

        Args:
            output_dir (Path): the directory in which the patches are saved
        """
        groups = self.df.groupby("setting")
        for setting_idx, group in groups:
            s = self.settings[setting_idx]
            self._export_patches_for_setting(
                group, output_dir, s.slide_path, s.level, s.patch_size, s.loader
            )

    def description(self):
        """ Returns basic summary of patchset

        returns the labels and the total number of patches of each label
    
        """
        labels = np.unique(self.df.label)
        sum_totals = [np.sum(self.df.label == label) for label in labels]
        return labels, sum_totals

    def _export_patches_for_setting(
        self,
        frame: pd.DataFrame,
        output_dir: Path,
        slide_path: Path,
        level: int,
        patch_size: int,
        loader: Loader,
    ):
        """Creates all the patches for an individual PatchSetting"""
        def get_output_dir_for_label(label: str) -> Path:
            label_str = invert(loader.labels)[label]
            label_dir = output_dir / label_str
            return label_dir

        def make_patch_path(x: int, y: int, label: int) -> Path:
            filename = f"{Path(slide_path).stem}-{x}-{y}-{level}-{patch_size}.png"
            label_dir = get_output_dir_for_label(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            return label_dir / filename

        def save_patch(region: Region, slide: SlideBase, filepath: Path) -> None:
            image = slide.read_region(region)
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), np.array(opencv_image))

        with loader.load_slide(slide_path) as slide:
            for row in frame.itertuples():
                filepath = make_patch_path(row.x, row.y, row.label)
                region = Region.make(row.x, row.y, patch_size, level)
                save_patch(region, slide, filepath)
