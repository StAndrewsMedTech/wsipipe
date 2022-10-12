from math import ceil

import numpy as np
from PIL import Image

from wsipipe.preprocess.patching import make_patchset_for_slide
from wsipipe.utils import pool2d
from wsipipe_eval.whole_slide_dataloader import get_pytorch_slide_data_loader
from wsipipe_eval.eval_model_on_slide import eval_model_on_slide


def to_heatmap(patches_df, class_name: str, border, jitter, patch_size, stride) -> np.array:
    patches_df.columns = [colname.lower() for colname in patches_df.columns]
    class_name = class_name.lower()

    # top left position is inclusive of border and jitter - need to remove to give core patch position
    adjust_top_left = ceil(border / 2) + jitter

    # find top left positions without border and jitter
    top_x = np.subtract(patches_df.x, adjust_top_left)
    left_y = np.subtract(patches_df.y, adjust_top_left)

    # find core patch size
    base_patch_size = patch_size - border - jitter

    pool_size = int(base_patch_size / stride)
    base_patch_size = stride

    # remove border and convert to column, row
    patches_df['column'] = np.divide(top_x, base_patch_size)
    patches_df['row'] = np.divide(left_y, base_patch_size)

    max_rows = int(np.max(patches_df.row)) + 1
    max_cols = int(np.max(patches_df.column)) + 1

    # create a blank thumbnail
    thumbnail_out = np.zeros((max_rows, max_cols))

    # for each row in dataframe set the value of the pixel specified by row and column to the probability in clazz
    for rw in patches_df.itertuples():
        thumbnail_out[int(rw.row), int(rw.column)] = getattr(rw, class_name)

    thumbnail_out = pool2d(thumbnail_out, pool_size, pool_size, 0, "avg")

    thumbnail_out = Image.fromarray(np.array(thumbnail_out * 255, dtype=np.uint8))

    return thumbnail_out
    

def eval_per_slide(slide_ps, slide_path, batch_size, transform, model, device, output_root):

    # save output
    slide_dir = output_root / slide_path.stem
    slide_dir.mkdir(parents=True, exist_ok=True)

    # create dataloader
    slide_dl = get_pytorch_slide_data_loader(slide_ps, batch_size, transform)   
    
    # - output features

    # - output probability dataframe
    patches_df = eval_model_on_slide(model, slide_dl, device, labels)
    patches_df.to_csv(slide_dir / "patch_results.csv", index=False)

    return patches_df


def to_heatmaps(patches_df: pd.DataFrame, patch_finder, labels):
    # - output heatmap
    for lab in labels:
        heatmap = to_heatmap(patches_df, class_name = lab, border = patch_finder.border, jitter = patch_finder.jitter, patch_size = patch_finder.patch_size, stride = patch_finder.stride)
        heatmap.save(slide_dir / ("heatmap_" + lab + ".png"))