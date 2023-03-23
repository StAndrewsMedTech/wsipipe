"""
This module creates the dataframe for the camelyon 17 dataset with the following columns:
    - The slide column stores the paths on disk of the whole slide images.
    - The annotation column records a path to the annotation files.
    - The label column is the slide level label.
    - The tags column is blank for camelyon 16.
This assumes there is a folder on disk structured the same as downloading
from the camelyon grand challenge Camelyon 17 google drive:
https://camelyon17.grand-challenge.org/Data/
"""

from csv import reader
from pathlib import Path
import pandas as pd


def training(cam17_path: Path = Path("data", "camelyon17"), project_root: Path = None) -> pd.DataFrame:
    """ Create Camelyon 17 training dataset
    This function goes through the input directories for the training slides,
    and matches up the annotations and slides.
    It creates a dataframe with slide path with matching annotation path, and slide label.
    There is an empty tags column that is not used for this dataset
    Args:
        cam17_path (Path, optional): a path relative to the project root that is the location
            of the Camelyon 17 data. Defaults to data/camelyon17.
    Returns:
        df (pd.DataFrame): A dataframe with columns: slide, annotation, label and tags
    """
    # Set up the paths to the slides and annotations
    # For Camelyon17, data is split into centers also, need to combine data from all different centers and patients

    if project_root is None:
        dataset_root = Path(cam17_path) / "training"
    else:
        dataset_root = project_root / Path(cam17_path) / "training"
    annotation_dir = dataset_root / "lesion_annotations"
    center_0_annotations_dir = dataset_root / "lesion_annotations" / "center_0"
    center_2_annotations_dir = dataset_root / "lesion_annotations" / "center_2"
    center_3_annotations_dir = dataset_root / "lesion_annotations" / "center_3"
    center_4_annotations_dir = dataset_root / "lesion_annotations" / "center_4"
    center_0_slide_dir = dataset_root / "center_0"
    center_1_slide_dir = dataset_root / "center_1"
    center_2_slide_dir = dataset_root / "center_2"
    center_3_slide_dir = dataset_root / "center_3"
    center_4_slide_dir = dataset_root / "center_4"

    # all paths are relative to the project root if defined
    if project_root is None:
        annotation_paths = sorted(
            [p for p in annotation_dir.glob("*.xml")]
        )
        center_0_slide_paths = sorted(
            [p for p in center_0_slide_dir.glob("*.tif")]
        )
        center_1_slide_paths = sorted(
            [p for p in center_1_slide_dir.glob("*.tif")]
        )
        center_2_slide_paths = sorted(
            [p for p in center_2_slide_dir.glob("*.tif")]
        )
        center_3_slide_paths = sorted(
            [p for p in center_3_slide_dir.glob("*.tif")]
        )
        center_4_slide_paths = sorted(
            [p for p in center_4_slide_dir.glob("*.tif")]
        )

    else:
        annotation_paths = sorted(
            [p.relative_to(project_root) for p in annotation_dir.glob("*.xml")]
        )
        center_0_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_0_slide_dir.glob("*.tif")]
        )
        center_1_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_1_slide_dir.glob("*.tif")]
        )
        center_2_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_2_slide_dir.glob("*.tif")]
        )
        center_3_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_3_slide_dir.glob("*.tif")]
        )
        center_4_slide_paths = sorted(
            [p.relative_to(project_root) for p in center_4_slide_dir.glob("*.tif")]
        )

    # turn them into a data frame and pad with empty annotation paths
    # Combine all centers for one directory of all slides
    df = pd.DataFrame()
    df["slide"] = center_0_slide_paths + center_1_slide_paths + center_2_slide_paths + center_3_slide_paths + center_4_slide_paths

    # Assign annotations and labels (as camelyon17 isn't ordered by tumour/normal slides, this has to be manually
    # filtered
    with open('/data/ec259/camelyon17/raw/training/stage_labels.csv', 'r') as read_obj:

        csv_reader = reader(read_obj)
        header = next(csv_reader)

        if header is not None:
            patient_counter = 0
            node_counter = -1
            center_counter = 0
            annotations = []
            labels = []

            for row in csv_reader:
                if row[0].__contains__('.zip'):
                    continue
                else:
                    node_counter = node_counter + 1

                # if there is no annotation (no tumour), add empty path to annotations
                if row[1].__contains__('negative'):
                    annotations.append("")
                    labels.append("normal")
                else:
                    # get the annotation path for the related annotation for the current slide
                    path = Path(str(annotation_dir) + "/center_" + str(center_counter) + "/" + row[0][:-4] + ".xml")
                    annotations.append(path)
                    labels.append("tumor")

                if node_counter == 5:
                    node_counter = 0
                    patient_counter = patient_counter + 1
                    if patient_counter == 19:
                        center_counter = center_counter + 1

        df["annotation"] = annotations
        df["label"] = labels
        df["tags"] = ""
    return df
