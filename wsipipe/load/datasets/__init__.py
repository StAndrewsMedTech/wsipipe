"""
Loaders specify formats for a particular dataset.
    - Specify the slide and annotation type for a dataset
    - Specify the labels and grouping of labels for a dataset

"""

from .loader import *
from .registry import *
from .camelyon16 import *
from .stripai import *