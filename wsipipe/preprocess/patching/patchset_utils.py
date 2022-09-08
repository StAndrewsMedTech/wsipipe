import itertools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from wsipipe.load.datasets import Loader
from wsipipe.preprocess.tissue_detection import TissueDetector
from wsipipe.preprocess.patching.patch_finder import PatchFinder
from wsipipe.preprocess.patching.patchset import PatchSet, PatchSetting


def make_patchset_for_slide(
    slide_path: Path,
    annot_path: Path,
    loader: Loader,
    tissue_detector: TissueDetector,
    patch_finder: PatchFinder,
    project_root: Path = Path('/')
) -> PatchSet:

    with loader.load_slide(project_root / slide_path) as slide:
        annotations = loader.load_annotations(project_root / annot_path)
        labels_shape = slide.dimensions[patch_finder.labels_level].as_shape()
        scale_factor = 2**patch_finder.labels_level
        labels_image = annotations.render(labels_shape, scale_factor)
        tissue_mask = tissue_detector(slide.get_thumbnail(patch_finder.labels_level))
        labels_image[~tissue_mask] = 0
        df, level, size = patch_finder(
            labels_image, slide.dimensions[patch_finder.patch_level]
        )
        df["setting"] = 0  # set all the patches to reference the first patchsetting
        patchset = PatchSet(df, [PatchSetting(level, size, slide_path, loader)])
        return patchset


def make_and_save_patchsets_for_dataset(
    dataset: pd.DataFrame,
    loader: Loader,
    tissue_detector: TissueDetector,
    patch_finder: PatchFinder,
    output_dir: Path,
    project_root: Path = Path('/')    
) -> List[PatchSet]:

    patchsets = []
    for row in dataset.itertuples():
        patchset_path = output_dir / Path(row.slide).stem
        if patchset_path.is_dir():
            patchset = PatchSet.load(patchset_path)
        else:
            patchset = make_patchset_for_slide(
                row.slide, row.annotation, loader, tissue_detector, patch_finder, project_root
            )
            patchset.save(patchset_path)
        patchsets.append(patchset)

    return patchsets


def load_patchsets_from_directory(patchsets_dir: Path):
    patchset_dir_list = [x for x in patchsets_dir.iterdir() if x.is_dir()]
    patchset_list = [PatchSet.load(p) for p in patchset_dir_list]
    return patchset_list


def combine(patchsets: List[PatchSet]) -> PatchSet:
    # compute and apply the settings index offset
    # offset is equal to the size of the settings object up to this point
    offset = 0
    for ps in patchsets:
        ps.df["setting"] += offset  # todo: do we mind mutating ps and thus patchsets
        offset += len(ps.settings)

    # merge the data frames
    frames_list = [ps.df for ps in patchsets]
    combined_df = pd.concat(frames_list, axis=0, ignore_index=True)

    # merge the setting lists
    combined_settings = []
    for ps in patchsets:
        combined_settings.extend(ps.settings)

    return PatchSet(combined_df, combined_settings)


def visualise_patches_on_slide(ps: PatchSet, vis_level, project_root: Path = Path('/')):
    assert len(ps.settings) == 1, "The input patch set contains patches from more than one slide."
    slide_settings = ps.settings[0]

    def convert_ps_to_thumb_level(ps, thumb_lev):
        ps_df = ps.df.copy()
        ps_df.x = ps_df.x.divide(2 ** thumb_lev).astype(int)
        ps_df.y = ps_df.y.divide(2 ** thumb_lev).astype(int)
        thumb_patch_size = slide_settings.patch_size // 2 ** thumb_lev
        return PatchSet(ps_df, [PatchSetting(slide_settings.level, thumb_patch_size,slide_settings.slide_path, slide_settings.loader)])

    def create_visualisation_frame(ps_in):
        vis_frame = ps_in.df
        # TODO: ps.settings[0] as only one settings is there a neater way to do this
        vis_frame["x2"] = vis_frame.x.add(ps_in.settings[0].patch_size)
        vis_frame["y2"] = vis_frame.y.add(ps_in.settings[0].patch_size)
        return vis_frame

    with slide_settings.loader.load_slide(project_root / slide_settings.slide_path) as slide:
            thumb = slide.get_thumbnail(vis_level)
        
    thumb = Image.fromarray(np.array(thumb, dtype=np.uint8))
    ps_out = convert_ps_to_thumb_level(ps, vis_level)
    vis_frame = create_visualisation_frame(ps_out)

    thumbdraw = ImageDraw.Draw(thumb) 
    for row in vis_frame.itertuples():
        thumbdraw.rectangle([row.x, row.y, row.x2, row.y2], fill=None, outline='black', width=1)
        
    return thumb