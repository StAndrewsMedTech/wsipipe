from typing import Callable

import numpy as np
import pandas as pd

from wsipipe.preprocess.patching import PatchSet


def simple_random(class_df: pd.DataFrame, sum_totals: int) -> pd.DataFrame:
    class_sample = class_df.sample(n=sum_totals, axis=0, replace=False)
    return class_sample

def simple_random_replacement(class_df: pd.DataFrame, sum_totals: int) -> pd.DataFrame:
    class_sample = class_df.sample(n=sum_totals, axis=0, replace=True)
    return class_sample


def weighted_random(class_df: pd.DataFrame, sum_totals: int) -> pd.DataFrame:
    class_df = class_df.assign(freq=class_df.groupby('slide_idx')['slide_idx'].transform('count').tolist())
    class_df = class_df.assign(weights= np.divide(1, class_df.freq))
    class_sample = class_df.sample(n=sum_totals, axis=0, replace=True, weights=class_df.weights)
    return class_sample


def balanced_sample(patches: PatchSet, num_samples: int, floor_samples: int = 1000, 
                    sampling_policy: Callable[[pd.DataFrame, int], pd.DataFrame] = simple_random) -> PatchSet:
    # note - if are working with data sets make sure that the indexes of the labels are
    # different for things that are different accross the datasets (eg. specific pathologies)
    # and the same for things that are the same (eg. tumor, normal)

    # work out how many of each type of patches you have in the index
    labels, sum_totals = patches.description()
    
    # find the count for the class that has the lowest count, so we have balanced classes
    n_patches = min(sum_totals)

    # limit the count for each class to the number of samples we want
    n_patches = min(n_patches, num_samples)

    # make sure that have a minimun number of samples for each class if available
    # classes with smaller that floor with remain the same
    n_patches = max(n_patches, floor_samples)
    sum_totals = np.minimum(sum_totals, n_patches)  # cap the number sample from each class to n_patches

    # sample n patches
    sampled_patches = pd.DataFrame(columns=patches.df.columns)
    for idx, label in enumerate(labels):
        class_df = patches.df[patches.df.label == label]
        class_sample = sampling_policy(class_df, sum_totals[idx])
        sampled_patches = pd.concat((sampled_patches, class_sample), axis=0)

    return PatchSet(sampled_patches, patches.settings)
