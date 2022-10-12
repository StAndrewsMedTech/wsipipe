import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

from wsipipe.load.slides import Region
from wsipipe.preprocess.patching import PatchSet

def get_val_from_settings(ps, val):
    """ Gets all of a single setting in a PatchSettings """
    slidevals = [getattr(st, val) for st in ps.settings]
    slidevals = np.unique(slidevals)
    return slidevals

def check_single_slide(ps):
    """ Checks if all PatchSettings are from single slide """
    slidepaths = get_val_from_settings(ps, "slide_path")
    nslidepaths = len(slidepaths)
    slidecheck = nslidepaths == 1
    return slidecheck

def create_single_slide_df(ps):
    """ If all PatchSettings are from single slide creates one array with x,y,label,patchsize,level """
    assert check_single_slide(ps), "The input patch set contains patches from more than one slide."
    new_df = pd.DataFrame(index=range(ps.df.shape[0]), columns=['x','y','label','patch_size','level'])
    for row in ps.df.reset_index().itertuples():
        row_settings = ps.settings[row.setting]
        new_df.iloc[row.Index] = [row.x, row.y, row.label, row_settings.patch_size, row_settings.level]
    return new_df

def on_init(ps):
    assert check_single_slide(ps), "The input patch set contains patches from more than one slide."
    slidepath = get_val_from_settings(ps, "slide_path")[0]
    patch_df = create_single_slide_df(ps)
    loader = get_val_from_settings(ps, "loader")[0]
    slide = loader.load_slide(slidepath)
    return slide, patch_df

def get_single_patch(idx, patch_df, slide):
    def to_patch(p: tuple, slide) -> Image:
        region = Region.make(p.x, p.y, p.patch_size, p.level)
        image = slide.read_region(region)
        image = image.convert('RGB')
        return image
    patch_info = patch_df.iloc[idx]
    image = to_patch(patch_info, slide)
    label = patch_info.label
    return image, label