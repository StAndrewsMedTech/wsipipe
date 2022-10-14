"""
This module creates the dataframe for the camelyon 16 dataset with the follwing columns: 
    - The slide column stores the paths on disk of the whole slide images.
    - The annotation column records a path to the annotation files.
    - The label column is the slide level label.
    - The tags column is blank for camelyon 16.

This assumes there is a folder on disk structured the same as downloading 
from the camelyon grand challenge Camelyon 16 google drive:
https://camelyon17.grand-challenge.org/Data/ 

"""

from pathlib import Path

import pandas as pd

def training(cam16_path: Path = Path("data", "camelyon16"), project_root: Path = Path()) -> pd.DataFrame:
    """ Create Camleyon 16 training dataset

    This function goes through the input directories for the training slides, 
    and matches up the annotations and slides. 
    It creates a dataframe with slide path with matching annotation path, and slide label.
    There is an empty tags column that is not used for this dataset 

    Args:
        cam16_path (Path, optional): a path relative to the project root that is the location 
            of the Camelyon 16 data. Defaults to data/camelyon16.
    Returns:
        df (pd.DataFrame): A dataframe with columns slide, annotation, label and tags
    """
    # set up the paths to the slides and annotations
    dataset_root = project_root / cam16_path / "training"
    annotations_dir = dataset_root / "lesion_annotations"
    tumor_slide_dir = dataset_root / "tumor"
    normal_slide_dir = dataset_root / "normal"

    # all paths are relative to the project root
    annotation_paths = sorted(
        [p.relative_to(project_root) for p in annotations_dir.glob("*.xml")]
    )
    tumor_slide_paths = sorted(
        [p.relative_to(project_root) for p in tumor_slide_dir.glob("*.tif")]
    )
    normal_slide_paths = sorted(
        [p.relative_to(project_root) for p in normal_slide_dir.glob("*.tif")]
    )

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = tumor_slide_paths + normal_slide_paths
    df["annotation"] = annotation_paths + ["" for _ in range(len(normal_slide_paths))]
    df["label"] = ["tumor"] * len(tumor_slide_paths) + ["normal"] * len(
        normal_slide_paths
    )
    df["tags"] = ""

    return df


def testing(cam16_path: Path = Path("data", "camelyon16"), project_root: Path = Path()) -> pd.DataFrame:
    """ Create Camleyon 16 testing dataset
    
    This function goes through the input directories for the testing slides, 
    and matches up the annotations and slides. 
    It creates a dataframe with slide path with matching annotation path, and slide label.
    There is an empty tags column that is not used for this dataset 

    Args:
        cam16_path (Path, optional): a path relative to the project root that is the location 
            of the Camelyon 16 data. Defaults to data/camelyon16.
    Returns:
        df (pd.DataFrame): A dataframe with columns slide, annotation, label and tags
    """
    # set up the paths to the slides and annotations
    dataset_root = project_root / cam16_path / "testing"
    annotations_dir = dataset_root / "lesion_annotations"
    slide_dir = dataset_root / "images"

    # all paths are relative to the dataset 'root'
    slide_paths = sorted([p.relative_to(project_root) for p in slide_dir.glob("*.tif")])
    annotation_paths = sorted(
        [p.relative_to(project_root) for p in annotations_dir.glob("*.xml")]
    )

    # get the slide name
    slide_names = [p.stem for p in slide_paths]

    # search for slides with annotations, add the annotation path if it exists else add empty string
    slides_annotations_paths = []
    for name in slide_names:
        a_path = ""
        for anno_path in annotation_paths:
            if name in str(anno_path):
                a_path = anno_path
        slides_annotations_paths.append(a_path)

    # get the slide labels by reading the csv file
    csv_path = dataset_root / "reference.csv"
    label_csv_file = pd.read_csv(csv_path, header=None)
    slide_labels = label_csv_file.iloc[:, 1]

    # turn them into a data frame and pad with empty annotation paths
    df = pd.DataFrame()
    df["slide"] = slide_paths
    df["annotation"] = slides_annotations_paths
    df["label"] = slide_labels
    df["tags"] = ""

    return df
