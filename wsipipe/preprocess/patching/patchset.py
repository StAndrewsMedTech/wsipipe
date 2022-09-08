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
    level: int
    patch_size: int
    slide_path: Path  # not stored in the dataframe
    loader: Loader  # not stored in the dataframe

    def to_sdict(self):
        d = asdict(self)
        d["slide_path"] = str(self.slide_path)
        d["loader"] = self.loader.name
        return d

    @classmethod
    def from_sdict(cls, sdict: dict):
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
        path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path / "frame.csv", index=False)
        dicts = [s.to_sdict() for s in self.settings]
        with open(path / "settings.json", "w") as outfile:
            json.dump(dicts, outfile)

    @classmethod
    def load(cls, path: Path) -> "PatchSet":
        print(f"loading {path}")
        df = pd.read_csv(path / "frame.csv")
        with open(path / "settings.json") as json_file:
            settings = json.load(json_file)
            settings = [PatchSetting.from_sdict(s) for s in settings]
        return cls(df, settings)

    def export_patches(self, output_dir: Path) -> None:
        groups = self.df.groupby("setting")
        for setting_idx, group in groups:
            s = self.settings[setting_idx]
            self._export_patches_for_setting(
                group, output_dir, s.slide_path, s.level, s.patch_size, s.loader
            )

    def description(self):
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
