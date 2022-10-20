=====
Tutorial
=====

Some basics to get you started using wsipipe. Wsipipe is structured around:
    - datasets, which contain the details of where files are stored.
    - patchsets, which contain details of where patches are within an WSI.

Specifying slide and annotation information 
===========================================

To get started we need to define a set of data we are going to use.
A dataset is stored in a pandas DataFrame. 
Each row contains the information for a single slide.
The dataframe should have four columns, slide, annotation, label, and tags.
    - slide, contains the path the WSI.
    - annotation, contains a path to an annotation file.
    - label, contains a slide level label.
    - tags, can contain any other information you want to store about the slide.
You can create these dataframes or read from disk. 
WSIpipe also has some datasets predefined, for example camelyon16.

If you have downloaded the camelyon16 data from and stored it in a local folder:
https://camelyon17.grand-challenge.org/Data/ .
It is assumed the local folder (path_to_local_folder) is structured in the dame way as the Camleyon16 google drive, 
that is it should contain two folders named training and testing.

The code to create the wsipipe dataset dataframe is::

    from wsipipe.datasets import camelyon16

    train_dset = camelyon16.training(cam16_path = path_to_local_folder)


We only want to use a few slides for the examples in this tutorial so we can cut down the size using sample_dataset.
For example if we want to randomly select 2 slides of each label category from the dataset::

    from wsipipe.datasets.dataset_utils import sample_dataset

    small_train_dset = sample_dataset(train_dset, 2)

As the dataset is just a pandas dataframe we can access information for an individual slide by specifying the row.::

    row = small_train_dset.iloc[0]

Specifying how to load a dataset
================================

Our dataset has now stored the location of the WSI, annotations and other information. 
Now we need to specify how these files are to be loaded as not all WSI formats and annotations
can be loaded using the same libraries.
This is done using dataset loader classes, each of which specifies how to load annotations and slides, 
as well as the allowable slide labels. 
A selection of slide and annotation loaders are included in wsipipe.
The Camleyon16 dataset loader class is specifed as:::

    from wsipipe.load.datasets.camelyon16 import Camelyon16Loader

    dset_loader = Camelyon16Loader()

Viewing a slide
===============

Now we have defined where the WSI files are and how to load them, we can open a slide and return 
the whole slide at a given level in the image pyramid as a numpy array. Depending on the size of 
the WSI it may not be possible to do this at the lowest levels (highest magnification)
of the image pyramid due to lack of memory. In the example we are extracting the thumbnail at 
level 5.::

    with dset_loader.load_slide(row.slide) as slide:
        thumb = slide.get_thumbnail(5)

This code returns a numpy array, if you want to for example display it as a PIL image in a jupyter notebook.::

    from wsipipe.utils import np_to_pil

    np_to_pil(thumb)

Viewing an annotation
=====================

We can also read and view the annotations, here we render them at level 5. 
The annotations for camleyon are read in as labels 1 or 2, 
in the code below they are mulitplied by 100 to make them visible when displayed.::

    from wsipipe.load.annotations import visualise_annotations

    labelled_image = visualise_annotations(
        row.annotation, 
        row.slide,
        dset_loader,
        5
    )
    np_to_pil(labelled_image*100)

Applying background subtraction
===============================

Often large parts of WSI are background that contain nothing of interest, 
therefore we want to split the background from the tissue so we know which are the areas of interest on the slide.
There different types of tissue detectors specfied in wsipipe. Here we use a basic Greyscale version.
Firstly we specify our tissue detector and define the parameters, then we apply it to a thumbnail of the WSI.
This returns a binary mask where True/1/white is tissue and False/0/black is background.::

    from wsipipe.preprocess.tissue_detection import TissueDetectorGreyScale
    
    tisdet = TissueDetectorGreyScale(grey_level=0.85)
    tissmask = tisdet(thumb)
    np_to_pil(tissmask)

We can also apply filters or morphological operations as part of the tissue detection.::

    from wsipipe.preprocess.tissue_detection import SimpleClosingTransform, GaussianBlur

    prefilt = GaussianBlur(sigma=2)
    morph = [SimpleOpeningTransform(), SimpleClosingTransform()]
    tisdet = TissueDetectorGreyScale(
        grey_level=0.75, 
        morph_transform = morph, 
        pre_filter = prefilt
    )
    tissmask = tisdet(thumb)
    np_to_pil(tissmask)

We can also visualise the mask overlaid on the thumbnail.::

    from wsipipe.preprocess.tissue_detection import visualise_tissue_detection_for_slide
    
    visualise_tissue_detection_for_slide(row.slide, dset_loader, 5, tisdet)


Creating a patchset for a slide
===============================

Next we define the location of patches to extract from the slide, which we refer to as a patchset. 
Here we specify we want to create 256 pixels patches on a regular grid with stride 256 pixels. 
The patches are extracted at level 0. This will be calculated based on thumbnails and annotations 
rendered at level 5.::

    from wsipipe.preprocess.patching import GridPatchFinder, make_patchset_for_slide

    patchfinder = GridPatchFinder(patch_level=1, patch_size=512, stride=512, labels_level=5)
    pset = make_patchset_for_slide(row.slide, row.annotation, dset_loader, tisdet, patchfinder)

The patchset is datafrom with the top left position and label for each patch, plus a settings object 
which stores information which is used for multiple patches such as the patch size and slide path. 
You can combine multiple settings within one patchset, so the dataframe also records which setting to apply to a patch.
We can then use the patchset to visualise the patches overlaid on the slide.::

    from wsipipe.preprocess.patching import visualise_patches_on_slide

    visualise_patches_on_slide(pset, vis_level = 5)

There is also a random patch finder available, which extracts a given number of patches at random locations
within the tissue area. 

Creating patchsets for a dataset
================================

We can also create patchsets for the whole dataset. This simply returns a list of patchsets for each slide in the dataset.::

    from wsipipe.preprocess.patching import make_patchsets_for_dataset

    psets_for_dset = make_patchsets_for_dataset(
        dataset = small_train_dset, 
        loader = dset_loader, 
        tissue_detector = tisdet, 
        patch_finder = patchfinder
    )

Saving and loading patchsets
============================

For large datasets, this can take a long time and a problem in one file can cause this not to complete. It is frustrating to 
have to remake the patchsets for all the other slides. Therefore there is also a function to save each patchset individually
as it makes them. When the function is rerun it then checks if the patchsets already exists, if so it skips creating it.
This function saves each patchset in a separate subdirectory of the output directory.::

    from wsipipe.preprocess.patching import make_and_save_patchsets_for_dataset

    psets_for_dset = make_and_save_patchsets_for_dataset(
        dataset = small_train_dset, 
        loader = dset_loader, 
        tissue_detector = tisdet, 
        patch_finder = patchfinder, 
        output_dir = path_to_pset_folder
    )

You can also load datasets created with the same folder structure.::

    from wsipipe.preprocess.patching import load_patchsets_from_directory

    psets_for_dset = load_patchsets_from_directory(patchsets_dir = path_to_pset_folder)

Combining patchsets
===================

You can combine multiple patchsets into one big patchset, for example to combine all the patchsets in a dataset.::

    from wsipipe.preprocess.patching import combine

    all_patches_in_dset = combine(psets_for_dset)

Sampling patchsets
==================

You can sample patches from a patchset, there are various samplers available that can be used to create 
balanced sets, weighted sets etc. The balanced sample will sample num_samples without replacement from each category.
If there are fewer than num_samples of one category it will sample the number of samples of the smallest 
category. If the smallest category is less than floor_samples, it will sample floor_samples
from the other categories and all the samples from the smallest category. The sampler returns a patchset.::

    from wsipipe.preprocess.sample import balanced_sample

    sampled_patches = balanced_sample(
        patches = all_patches_in_dset, 
        num_samples = 500, 
        floor_samples = 100
    )

Creating patches
================

Once you have a patchset (an individual slide, a combined patchset or a sampled patchset) 
it is simple to create the patches from it.::

    sampled_patches.export_patches(path_to_folder_for_patches)

You now have your patches ready for training the deep learning model of your choice.




