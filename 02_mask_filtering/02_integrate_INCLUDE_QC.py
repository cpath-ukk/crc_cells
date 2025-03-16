start = 0
end = 300

import os
import statistics
import numpy as np
from PIL import Image, ImageOps
import cv2
from scipy import ndimage
from wsi_colors import colors_COADREAD as colors
from tqdm import tqdm
from openslide import open_slide
Image.MAX_IMAGE_PIXELS = None

###PATHES
BASE_DIR = 'path/to/dir'

###FLAGS
FLAG_QC = True
FLAG_INCLUDE = False
FILL_ROI_WITH_TUMOR_FLAG = False

#CLASSES
LABEL_TUMOR = 1
LABEL_STROMA = 2

### PARAMETERS
THUMBNAIL = 'openslide' # If it should use tifffile or openslide. Use other string to just use jpg thumbnail
MPP_MODEL_QC = 1.5
M_P_S_MODEL_QC = 512


##############################################################
#DEFINE AND GENERATE DIRS
DIR_PATH_THUMBNAIL = BASE_DIR + 'tis_det_thumbnail/'  # overlay image from model inference
DIR_PATH_FILTER_MASK = BASE_DIR + 'mask_filter/'
DIR_PATH_QC_MASK = BASE_DIR + 'mask_qc/'
DIR_PATH_INCLUDE_MASK = BASE_DIR + 'mask_include/'

FINAL_MASK_SAVE_DIR = BASE_DIR + 'mask_final/'
FINAL_MASK_COLOR_SAVE_DIR = BASE_DIR + 'mask_final_color/'
FINAL_OVERLAY_SAVE_DIR = BASE_DIR + 'overlay_final/'


#FUNCTIONS
def make_color_map(mask, class_colors):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(1, len(class_colors) + 1):
        idx = mask == l
        r[idx] = class_colors[l - 1][0]
        g[idx] = class_colors[l - 1][1]
        b[idx] = class_colors[l - 1][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# SCRIPT
# Read mask images from the folder
try:
    os.mkdir(FINAL_MASK_SAVE_DIR)
    os.mkdir(FINAL_MASK_COLOR_SAVE_DIR)
    os.mkdir(FINAL_OVERLAY_SAVE_DIR)
except:
    print('Folder for filtered masks is already there.')

image_names = sorted(os.listdir(DIR_PATH_FILTER_MASK))


for image_name in tqdm(image_names [start:end], total=len(image_names)):
    #print(image_name)

    mask_filter = np.array(Image.open(DIR_PATH_FILTER_MASK + image_name)) # open slide mask
    he_target, wi_target = mask_filter.shape

    if FLAG_QC:
        try:
            mask_qc = np.array(Image.open(DIR_PATH_QC_MASK + image_name[:-16] + '_mask_qc.png'))
            mask_qc = cv2.resize(mask_qc, (wi_target, he_target),
                                      interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        except:
            mask_qc = None
            print("QC mask could not be loaded")


    if FLAG_INCLUDE:
        try:
            mask_include = np.array(Image.open(DIR_PATH_INCLUDE_MASK + image_name[:-24] + '_mask_include.png'))
            mask_include = cv2.resize(mask_include, (wi_target, he_target),
                                         interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        except:
            try:
                mask_include = np.array(Image.open(DIR_PATH_INCLUDE_MASK + image_name[:-24] + '_mask_include.png'))
                mask_include = cv2.resize(mask_include, (wi_target, he_target),
                                          interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            except:
                print("No mask INCLUDE")
                mask_include = None

    if FLAG_INCLUDE:
        if mask_include is not None:
            mask_filter = np.where(mask_include != 1, 0, mask_filter)
            if FILL_ROI_WITH_TUMOR_FLAG:
                mask_filter = np.where(np.logical_and(mask_include == 1, mask_filter == 0), 1, mask_filter)

    if mask_qc is not None:
        mask_qc = np.where(mask_qc == 5, 1, mask_qc) # This is necessary for pigmented melanomas that produce ART_AIR FPs
        mask_filter = np.where(mask_qc != 1, 0, mask_filter)

    result_mask = mask_filter

    Image.fromarray(result_mask).save(FINAL_MASK_SAVE_DIR + image_name [:-16] + '_mask_final.png')

    #Now making overlay with the thumbnail - Later with larger slide image (received through openslide or tifffile)
    result_mask_color = make_color_map(result_mask, colors)

    result_mask_color_image = Image.fromarray(result_mask_color)

    result_mask_color_image.save(FINAL_MASK_COLOR_SAVE_DIR + image_name[:-16] + '_mask_final_color.png')
    #print('finished cleanup')

    thumbnail = Image.open(DIR_PATH_THUMBNAIL + image_name [:-16] + '.jpg')

    result_mask_color_resize = result_mask_color_image.resize(thumbnail.size, Image.ANTIALIAS)

    overlay = cv2.addWeighted(np.array(thumbnail), 0.7, np.array(result_mask_color_resize), 0.3, 0)
    Image.fromarray(overlay).save(FINAL_OVERLAY_SAVE_DIR + image_name[:-16] + '_overlay_final.jpg')

    #Image.fromarray(overlay).show()