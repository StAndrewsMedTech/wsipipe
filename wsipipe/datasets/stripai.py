"""
This module creates the dataframe for the STRIP AI dataset with the following columns:
    - The slide column stores the paths on disk of the whole slide images
    - The annotation column records a string with the slide label 
    - The label column is the slide level label
    - The tags column contains the center and patient for each slide

This assumes there is a folder on disk structured the same as downloading
from the kaggle website
https://www.kaggle.com/competitions/mayo-clinic-strip-ai/data

"""

import os
from pathlib import Path
import pandas as pd

def convert_to_pyramids(mayo_path: Path = Path("data", "mayo-clinic-strip-ai"), project_root: Path = Path()):
    """ Create pyramids for whole slide images

    The whole slide images as downloaded only contain data at level 0, 
    no other levels are present. This can make it slow to access the slides.
    This function will run over all the slides in the the dataset and write
    out copies that contain a pyramid of levels.
    Files are written to folder experiments/pyramids/ 

    Args:
        mayo_path (Path, optional): a path relative to the project root that is the location 
            of the strip ai data. Defaults to data/mayo-clinic-strip-ai.

    """
    def convert(in_path, out_path):
        print(f"Converting {in_path}")
        os.system(f"vips tiffsave {in_path} {out_path} --compression=lzw --tile --tile-width=256 --tile-height=256 --pyramid")

    train_root = project_root / mayo_path / "train"
    test_root = project_root / mayo_path / "test"

    # train images
    for img_path in list((train_root / "train").glob("*.tif")):
        output_path = project_root / 'experiments' / 'pyramids' / "train" / img_path.name
        if not output_path.exists():
            convert(img_path, output_path)

    # test images
    for img_path in list((test_root / "test").glob("*.tif")):
        output_path = project_root / 'experiments' / 'pyramids' / "test" / img_path.name
        if not output_path.exists():
            convert(img_path, output_path)


def training(mayo_path: Path = Path("data", "mayo-clinic-strip-ai"), project_root: Path = Path()) -> pd.DataFrame:
    """ Create Strip AI training dataset
    
    This function goes through the input directories for the training slides, 
    and matches up the slide paths with infomation in the csv
    It creates a dataframe with slide path with matching slide label stored for both label and annotation.
    The tags column stores the patient id and center id. 

    Args:
        mayo_path (Path, optional): a path relative to the project root that is the location 
            of the stripai data. Defaults to data/mayo-clinic-strip-ai.
    Returns:
        df (pd.DataFrame): A dataframe with columns slide, annotation, label and tags
    """
    # set up the paths to the slides and annotations
    dataset_root = project_root / mayo_path / "train" / "train"
    labels_df = pd.read_csv(project_root / mayo_path / "train.csv")

    # turn them into a data frame and pad with empty annotation paths
    slidepaths = []
    annots = []
    labels = []
    tags = []
    imagelist =  list(dataset_root.glob("*.tif"))
    for sp in imagelist[0:10]:
        slidepaths.append(sp)
        imageid = sp.stem
        row = labels_df[labels_df.image_id == imageid]
        annot = row.label.iloc[0]
        label = row.label.iloc[0]
        tag = 'center' + str(row.center_id.iloc[0]) + "; patient " + str(row.patient_id.iloc[0])
        annots.append(annot)
        labels.append(label)
        tags.append(tag)

    df = pd.DataFrame()
    df["slide"] = slidepaths
    df["annotation"] = annots
    df["label"] = labels
    df["tags"] = tags

    return df

# convert_to_pyramids()