Reference of provided code (alpha-version)

NB! We provide raw working scripts used for the development and implementation.

NB! Most of the experiments and scripts were run in anaconda environment
using Ubuntu 22.04. We recommend using this setup.

##01_WSI_inference
This is an inference script to process the whole-slide images (WSI).

Three versions are available: 1) for main segmentation algorithm (uses
slides and tissue detection masks as input), 2) for subtyping/supervised pixel-wise
segmentation algorithm (uses multi-class tissue segmentation mask from main
algorithm as input + slides), 3) same as 2 but for classifiers based on
UNI and Prov-GigaPath feature extractors (original model checkpoints from
hugging face and trained classifiers upon features are necessary).

NB! We provide the script for review that is based on the openslide (v.3.4.1
or later) library.

-wsi_tis_detect.py: script to run isolated tissue vs background segmentation
(based on the segmentation algorithm from GrandQC tool - manuscript in review,
the tool will be open-sourced upon publications). For custom tissue detectors,
this script can be ignored and tissue detection map can be used in main.py from
other sources.
-main.py: script to start inference using a trained pixel-wise segmentation model.
Needs tissue mask as input that should be stored in outputs folder.

-run.sh: bash script to pipe the tissue segmentation and artifact detection
steps (can be ignored by custom tissue detection pipelines)

-wsi_colors.py: script where color scheme to be used is defined
-wsi_maps.py: script with function to make overlay of segmentation
mask on the original WSI
-wsi_process.py: script with processing pipeline of WSI, output:
segmentation mask (class codings), segmentation mask (RGB corresponding to
wsi_colors.py defined scheme)
-wsi_slide_info.py: pipeline to retrieve WSI metadata necessary for inference.
-wsi_tis_detect_helper_fx.py: helper functions for tissue segmentation script
(wsi_tis_detect.py)

The output txt file with processed slides will be saved.
For subtyping versions it will include subtype areas and final slide classification.

##02_mask_filtering
Scripts for filtering of multi-class segmentation masks, e.g. to remove the
artificially change regions, to remove tumor stroma parts without immediately
adjacent tumor, etc.

##03_generate_nuclei_maps
Main script for generating geoJSON files containing coordinates of cell
nuclei and their classification. The script uses the segmentation masks
of tumor and tumor stroma region from previous steps.

##04_analyze_results
Script for analysing results, extracting single cell global and regional
parameters, building heatmaps.
