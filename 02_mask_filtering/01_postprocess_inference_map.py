'''
Now the final mask is the dilated, smoothed mask of tumor tissue:
- Small fragments of tissue are removed from tissue map
- These small fragments are zeroed for tumor pixels in tumor map
- 100 pixels of each side of the slide mask are zeroed for tumor - to prevent false positives from glass edge
- Tumor regions are inflated and smoothed.
- After inflation - small tumor regions that do not confluence with each other (therefore, likely, false positive)
are zeroed leaving confluent bulk.
- Stroma only left in the inflated tumor regions
- All other classes' pixels are zeroed, so that final mask contains only tumor pixels and stroma pixels (latter
only in inflated tumor regions)
'''

'''
#File number in folder to start and end. You can provide very large number for 'end' not to look
how many files exactly you have.
'''
start = 0
end = 200

import os
import statistics
import numpy as np
from PIL import Image, ImageOps
import cv2
from scipy import ndimage
from wsi_colors import colors_LUNG as colors
from tqdm import tqdm
from openslide import open_slide

Image.MAX_IMAGE_PIXELS = None

###PATHES
BASE_DIR = 'path/to/output/folder'
SLIDE_DIR = 'path/to/slide/folder/'

DIR_PATH_MASK = BASE_DIR + 'mask_inf/'  # png 1-layer pixel-wise class maps
DIR_PATH_THUMBNAIL = BASE_DIR + 'tis_det_thumbnail/'  # overlay image from model inference
DIR_PATH_TIS_MASK = BASE_DIR + 'tis_det_mask/'
FILTER_MASK_SAVE_DIR = BASE_DIR + 'mask_filter/'
FILTER_MASK_COLOR_SAVE_DIR = BASE_DIR + 'mask_filter_color/'
FILTER_OVERLAY_SAVE_DIR = BASE_DIR + 'overlay_filter/'

LABEL_TUMOR = 1
LABEL_STROMA = 2

### PARAMETERS
DILATION_SIZE = 100 #size to dilate tumor regions to leave only stroma peritumoral
BORDER_WIDTH = 100 #Margin for the whole slide to zero evtl. tumor pixels due to slide edge artifacts
THRESHOLD_TISSUE = 50000 #Number of pixels for tissue map (object smaller than these will be removed from the tissue map)
DILATION_BLUR_KERNEL = 9 # Leave this unchanged
INFL_TUMOR_THRESHOLD = 50000 #Threshold to remove small tumor regions in inflated tumor map which mostly originate from small false
# positive misclassifications. Otherwise they will connect themselves to larger tumor bulk after inflation.
THUMBNAIL = 'openslide' # If it should use tifffile or openslide. Use other string to just use jpg thumbnail
OVERLAY_FACTOR = 10 # reduction factor of the overlay compared to dimensions of original WSI
MPP_MODEL_1 = 1.0
M_P_S_MODEL_1 = 512
##############################################################

SLIDE_OVERLAY_FLAG = False


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
    os.mkdir(FILTER_MASK_SAVE_DIR)
    os.mkdir(FILTER_MASK_COLOR_SAVE_DIR)
    os.mkdir(FILTER_OVERLAY_SAVE_DIR)
except:
    print('Folder for filtered masks is already there.')

image_names = sorted(os.listdir(DIR_PATH_MASK))


for image_name in tqdm(image_names [start:end], total=len(image_names)):
    #print(image_name)

    mask = np.array(Image.open(DIR_PATH_MASK + image_name)) # open slide mask
    tissue_mask = np.array(Image.open(DIR_PATH_TIS_MASK + image_name[:-8] + 'MASK.png')) # open tissue mask of the slide

    if THUMBNAIL == 'openslide':
        path_slide = os.path.join(SLIDE_DIR, image_name[:-9])
        slide = open_slide(path_slide)
        mpp = float(slide.properties["openslide.mpp-x"])
        p_s = int(MPP_MODEL_1 / mpp * M_P_S_MODEL_1) #original p_s for inference at slide MPP

        # Extract and save dimensions of level [0]
        dim_l0 = slide.level_dimensions[0]
        w_l0 = dim_l0[0]
        h_l0 = dim_l0[1]

        # Calculate number of patches to process
        patch_n_w_l0 = int(w_l0 / p_s)
        patch_n_h_l0 = int(h_l0 / p_s)

        # now get size of padded region (buffer) at Model MPP
        buffer_right_l = int((w_l0 - (patch_n_w_l0 * p_s)) * mpp / MPP_MODEL_1)
        buffer_bottom_l = int((h_l0 - (patch_n_h_l0 * p_s)) * mpp / MPP_MODEL_1)
        # firstly bottom
        buffer_bottom = np.full((buffer_bottom_l, mask.shape[1]), 0)
        temp_image = np.concatenate((mask, buffer_bottom), axis=0)
        # now right side
        temp_image_he, temp_image_wi = temp_image.shape  # width and height
        buffer_right = np.full((temp_image_he, buffer_right_l), 0)
        mask = np.concatenate((temp_image, buffer_right), axis=1).astype(np.uint8)
        slide.close()

    else:
        continue

    '''
    reduce the dimensions of the mask to work easier
    '''
    mask_red = cv2.resize(mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

    '''
    resize the dimensions of the tissue mask to mask_red
    '''

    tissue_mask_red = cv2.resize(tissue_mask, (mask_red.shape[1], mask_red.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    '''
    Create only tumor mask from the mask_red
    '''
    mask_red_tumor = np.where(mask_red != 1, 0, mask_red)

    #ImageOps.autocontrast(Image.fromarray(mask_red_tumor)).show()
    #ImageOps.autocontrast(Image.fromarray(tissue_mask_red)).show()

    # Label connected components
    labeled_mask, num_features = ndimage.label(tissue_mask_red == 0)  # Labeling tissue (0) in reduced tissue mask

    # Compute sizes of connected components in the reduced tissue mask (detecting small structures to be removed)
    sizes = ndimage.sum(tissue_mask_red == 0, labeled_mask, range(num_features + 1))

    # Create a mask for all features larger than the threshold
    mask_size = sizes < THRESHOLD_TISSUE
    remove_pixel = mask_size[labeled_mask]

    # Replace small objects with background (1)
    tissue_mask_red[remove_pixel] = 1

    #ImageOps.autocontrast(Image.fromarray(tissue_mask_red)).show()

    '''
    Now based on updated tissue map, remove corresponding eventual tumor pixels in the tumor map that originate in
    small tissue objects that were removed in updated tissue map'''
    mask_red_tumor = np.where(tissue_mask_red != 0, 0, mask_red_tumor)

    # Setting the border pixels to zero in reduced tumor map. This is being done to remove potential false positive tumor pixels
    # due to the glass edge artifacts.
    mask_red_tumor[:BORDER_WIDTH, :] = 0  # Top border
    mask_red_tumor[-BORDER_WIDTH:, :] = 0  # Bottom border
    mask_red_tumor[:, :BORDER_WIDTH] = 0  # Left border
    mask_red_tumor[:, -BORDER_WIDTH:] = 0  # Right border

    '''
    Now dilation of the tumor to remove unnecessary stroma
    '''
    #Create binary mask of tumor regions from the reduced tumor mask
    binary_mask = (mask_red_tumor == 1).astype('uint8') * 255

    # Dilation for enlarging tumor regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATION_SIZE, DILATION_SIZE))
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)

    smoothed = cv2.medianBlur(dilated, DILATION_BLUR_KERNEL)

    _, binary_smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    final_mask = (binary_smoothed == 255).astype('uint8')

    #ImageOps.autocontrast(Image.fromarray(final_mask)).show()

    # Label connected components to identify small, now inflated, not connected tumor regions - mostly due
    # to false positive small regions.
    labeled_mask, num_features = ndimage.label(final_mask == 1)  # Labeling tissue (0)

    # Compute sizes of connected components
    sizes = ndimage.sum(final_mask == 1, labeled_mask, range(num_features + 1))

    # Create a mask for all features larger than the threshold
    mask_size = sizes < INFL_TUMOR_THRESHOLD

    remove_pixel = mask_size[labeled_mask]

    # Replace small inflated tumor objects with background (1) pixels
    final_mask[remove_pixel] = 0

    #ImageOps.autocontrast(Image.fromarray(final_mask)).show()

    '''
    Now postprocessing tumor stroma - leaving only tumor stroma in inflated tumor regions 
    final_mask - inflated tumor regions
    mask_red_tumor - exact tumor regions with zeroed borders
    tissue_mask_red - corrected tissue mask
    mask_red - reduced full mask with all classes
    '''

    '''
    1. Zero all pixel values outside of inflated tumor regions (final_mask)
    Otherwise same pixel coding scheme
    '''
    result_mask = np.where(final_mask == 0, 0, mask_red)

    '''
    2. Zero all non-tumor and non-tumor stroma pixels.
    These is for other classes' pixels that are within the inflated tumor mask
    '''
    result_mask = np.where(np.logical_and(result_mask != LABEL_TUMOR, result_mask != LABEL_STROMA), 0, result_mask)

    Image.fromarray(result_mask).save(FILTER_MASK_SAVE_DIR + image_name [:-4] + '_filter.png')

    #Now making overlay with the thumbnail - Later with larger slide image (received through openslide or tifffile)
    result_mask_color = make_color_map(result_mask, colors)

    result_mask_color_image = Image.fromarray(result_mask_color)

    result_mask_color_image.save(FILTER_MASK_COLOR_SAVE_DIR + image_name[:-4] + '_filter_color.png')
    #print('finished cleanup')

    if SLIDE_OVERLAY_FLAG:
        path_slide = os.path.join(SLIDE_DIR, image_name[:-9])
        slide = open_slide(path_slide)
        w_l0, h_l0 = slide.level_dimensions[0]
        slide_reduced = slide.get_thumbnail((w_l0 / OVERLAY_FACTOR, h_l0 / OVERLAY_FACTOR))
        heatmap_temp = result_mask_color_image.resize(slide_reduced.size, Image.ANTIALIAS)
        overlay = cv2.addWeighted(np.array(slide_reduced), 0.7, np.array(heatmap_temp), 0.3, 0)
        Image.fromarray(overlay).save(FILTER_OVERLAY_SAVE_DIR + image_name[:-4] + '_filter_overlay.jpg')
    else:
        thumbnail = Image.open(DIR_PATH_THUMBNAIL + image_name [:-9] + '.jpg')

        result_mask_color_resize = result_mask_color_image.resize(thumbnail.size, Image.ANTIALIAS)

        overlay = cv2.addWeighted(np.array(thumbnail), 0.7, np.array(result_mask_color_resize), 0.3, 0)
        Image.fromarray(overlay).save(FILTER_OVERLAY_SAVE_DIR + image_name[:-4] + '_filter_overlay.jpg')