from collections import namedtuple
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


def get_global_features(img_grey: np.array, grey_lev: float, feat_list: List[str]) -> pd.DataFrame:
    """ Create features based on whole slide properties of a given threshold
    feat list is a selection of area_ratio, avg_prob_class, avg_prob_tissue, tissue_area
    """
    threshold_mask = img_grey > grey_lev
    threshold_area = np.sum(threshold_mask)

    prob_mask = np.where(threshold_mask, img_grey, 0)
    prob_sum = np.sum(prob_mask)

    tissue_area = np.sum(img_grey > 0)
    if tissue_area > 0:
        area_ratio = threshold_area / tissue_area
        avg_prob_class = prob_sum / threshold_area
        avg_prob_tissue = prob_sum / tissue_area
    else:
        area_ratio = 0
        avg_prob_class = 0
        avg_prob_tissue = 0

    GlobFeat = namedtuple("GlobFeat", {"area_ratio", "avg_prob_class", "avg_prob_tissue", "tissue_area"})
    gf = GlobFeat(area_ratio, avg_prob_class, avg_prob_tissue, tissue_area)

    output_array = np.empty((1, len(feat_list)))
    colnames = []
    for idx, ft in enumerate(feat_list):
        output_array[0, idx] = getattr(gf, ft)
        colnames.append(ft + "_" + str(grey_lev))           
    
    output_df = pd.DataFrame(output_array, columns = colnames)

    return output_df


def get_region_features(img_grey: np.array, grey_lev: float, nregs: int, feats: List[str]) -> pd.DataFrame:
    """ Get list of properties of top n regions
    features can be any of the following output by regionprops function in scikit image:
    num_pixels, area, area_bbox, area_convex, area_filled, axis_major_length, axis_minor_length,
    eccentricity, equivalent_diameter_area, euler_number, extent, feret_diameter_max, 
    intensity_max, intensity_mean, intensity_min, perimeter, perimeter_crofton, solidity.
    Plus aspect ratio calculated from bounding box of region
    """
    # threshold image
    threshold_mask = img_grey > grey_lev
    # get regions
    labelled_img = label(threshold_mask, background=0)
    # measure regions
    reg_props = regionprops(labelled_img, intensity_image=img_grey)
    # get area for each region
    img_areas = [reg.area for reg in reg_props]
    # get labels for each region
    img_label = [reg.label for reg in reg_props]
    # sort in descending order
    toplabels = [x for _, x in sorted(zip(img_areas, img_label), reverse=True)]
    # get top nregs labels
    toplabels = toplabels[0:nregs]

    # output array
    slide_fts = np.empty((1, nregs * len(feats)))
    # per region add to store 
    for regidx, regno in enumerate(toplabels):
        # labels start from 1 need to subtract 1 for zero indexing
        reg = reg_props[regno - 1]
        for ftidx, ft in enumerate(feats):
            ftcnt = regidx * len(feats) + ftidx
            slide_fts[0, ftcnt] = reg[ft]
            if ft == "aspect_ratio":
                bbox = reg.bbox
                slide_fts[idx, ftcnt] = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])

    colnames = []
    for reg in range(nregs):
        for ft in feats:
            colnames.append(ft + "_" + str(reg) + "_" + str(grey_lev))
    
    output_df = pd.DataFrame(slide_fts, columns = colnames)

    return output_df