"""
Datasets contain information on sets of data, e.g file locations, number of slides, labels etc 
A dataset is a dataframe with columns slide, annotation, label and tags
    - slide contains WSI path
    - annotation contains path to annotation file or slide label
    - label contains slide level labels
    - tags is any other infomation about the slide (multiple pieces of data are separated by semi colons).
"""

from .dataset_utils import *