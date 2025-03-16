from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import json
from openslide import OpenSlide
from shapely.geometry import shape, box, GeometryCollection, Polygon, MultiPolygon
from shapely.affinity import scale
import numpy as np
import zarr
from skimage.io import imread
import rasterio.features
import rtree
import os
import pandas as pd
import csv
import math
import re

Image.MAX_IMAGE_PIXELS = None

# Define variables

input_WSI_folder = '/media/vmitchell/ssd8tb/TCGA_COADREAD/'
input_mask_folder = '/media/vmitchell/ssd8tb/TCGA_COADREAD_MAPS/mask_filter/'
input_geojson_folder = '/media/vmitchell/ssd8tb/TCGA_COADREAD_NUCLEI/'
input_qc_folder = '/media/vmitchell/ssd8tb/TCGA_COADREAD_QC/tis_det_mask/'
input_thumbnail_folder = '/media/vmitchell/ssd8tb/TCGA_COADREAD_MAPS/tis_det_thumbnail/'
include_image_folder = ''

output_path = '/media/vmitchell/d9b4230a-e8fa-4b83-9965-fe2293d30b6a/TCGA_CSV_Changed_cells/'
output_path_global_csv = output_path + 'csv/' + 'image_level_data.csv'
csv_file_exists = os.path.exists(output_path_global_csv)

os.makedirs(output_path + 'csv/', exist_ok=True)
os.makedirs(output_path + 'heatmaps/', exist_ok=True)

# Wether you are importing a QC or include image
use_qc = True
use_include = False

patch_size = 4000
min_mask_amount = 0.2

# Blur factor gets calculated from the patch size, to remain consistent
blur_factor = patch_size // 50

# How much the thumbnail should be darkened in the heatmap images
darkening_factor = 0.5

original_mpp = 1

# Color map used for the heatmap
heatmap_type = 'rainbow' # good alternatives are 'jet' or 'turbo'

# Colors used in the mask image to label the tissue
background_color = 0
tumor_color = 1
stroma_color = 3

qc_color = [2, 3, 4, 5, 6] # Must be a list
include_color = 1

# How much of the patches the numbers should take up in the patch label image
display_number_size = 0.75

# Define functions
def find_file_containing_string(folder_path, search_string):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            if search_string in file_path:
                return True
    return False

def output_exists(filename, output_path):
    for file in os.listdir(output_path + 'csv/'):
        match = re.search(r'^(.*?)_\d+_\d+_\d+_\d+_patch_data$', os.path.splitext(file)[0])
        if match and match.group(1) == os.path.splitext(filename)[0]:
            return True
    return False

def generate_patches(wsi_width, wsi_height, patch_size):
    patches = []
    for y in range(0, wsi_height, patch_size):
        for x in range(0, wsi_width, patch_size):
            # Ensure xmax and ymax don't exceed the image dimensions
            xmax = min(x + patch_size, wsi_width)
            ymax = min(y + patch_size, wsi_height)
            
            patches.append([x, y, xmax, ymax])
    return patches

def extract_polygons(binary_array):
    # Extract shapes from the binary array
    results = list(rasterio.features.shapes(binary_array.astype(np.int16), mask=(binary_array == 1)))
    polygons = [shape(geom) for geom, value in results if value == 1.0]
    return polygons

def image_to_geometry_collection(chunk, color):
    mask = np.array(chunk)
    binary_array = (mask == color).astype(int)

    polygons = extract_polygons(binary_array)

    geometry_collection = GeometryCollection(polygons)

    return geometry_collection

def has_significant_area(geometry_collection_tumor, geometry_collection_stroma, patch_size, min_mask_amount):
    total_area = patch_size * patch_size
    significant_area = total_area * min_mask_amount
    tumor_area = sum(polygon.area for polygon in geometry_collection_tumor.geoms)
    stroma_area = sum(polygon.area for polygon in geometry_collection_stroma.geoms)

    return tumor_area + stroma_area >= significant_area

def cut_image(image, patches):
    chunks = []

    for (xmin, ymin, xmax, ymax) in patches:
        # Crop the image based on the coordinates
        chunk = image.crop((xmin, ymin, xmax, ymax))

        # Check and adjust the size of the chunk if necessary
        if chunk.size != (xmax - xmin, ymax - ymin):
            chunk = chunk.resize((xmax - xmin, ymax - ymin))

        chunks.append(chunk)

    return chunks

def get_downscale_factor(input_WSI, original_mpp=1):
    try:
        wsi = OpenSlide(input_WSI)
        wsi_width, wsi_height = wsi.level_dimensions[0]
        try:
            mpp = float(wsi.properties["openslide.mpp-x"])
        except:
            mpp = 0.2305
    except:        
        slide_tiff = imread(input_WSI, aszarr=True)
        wsi = zarr.open(slide_tiff, mode='r')[0]
        wsi_height, wsi_width = wsi.shape
        mpp = 0.2325
    p_s = int(original_mpp / mpp * 512)
    patch_numbers = wsi_width / p_s
    proper_width = patch_numbers * 128
    mask_downscale_factor = wsi_width / proper_width 
    return mask_downscale_factor, wsi_width, wsi_height, mpp

def scale_geometry_collection(geom_collection, mask_downscale_factor):
    scaled_geometries = []
    for geom in geom_collection.geoms:
        scaled_geometries.append(scale(geom, mask_downscale_factor, mask_downscale_factor, origin=(0, 0)))
    return GeometryCollection(scaled_geometries)

def get_significant_area_indices(poly_tumor_map_patches, poly_stroma_map_patches, patch_size, min_mask_amount=0.05):
    significant_indices = []
    
    for idx, (geometry_collection_tumor, geometry_collection_stroma) in enumerate(zip(poly_tumor_map_patches, poly_stroma_map_patches)):
        if has_significant_area(geometry_collection_tumor, geometry_collection_stroma, patch_size, min_mask_amount):
            significant_indices.append(idx)
    
    return significant_indices

def include_only_areas(tissue_map, mask_map, mask_color):
    # Step 1: Scale the qc_map image to the size of tissue_map using nearest neighbor interpolation
    mask_map_resized = mask_map.resize(tissue_map.size, Image.NEAREST)
    
    # Convert images to numpy arrays for faster operations
    tissue_map_arr = np.array(tissue_map)  # Explicitly create a writable copy
    mask_map_resized_arr = np.asarray(mask_map_resized)
    
    # Where qc_map_resized is black, set the corresponding pixel in tissue_map to black
    tissue_map_arr[mask_map_resized_arr != mask_color] = 0

    # Convert the result back to an image
    result_img = Image.fromarray(tissue_map_arr)
    return result_img

def exclude_only_areas(tissue_map, mask_map, mask_color_list):
    # Step 1: Scale the qc_map image to the size of tissue_map using nearest neighbor interpolation
    mask_map_resized = mask_map.resize(tissue_map.size, Image.NEAREST)
    
    # Convert images to numpy arrays for faster operations
    tissue_map_arr = np.array(tissue_map)  # Explicitly create a writable copy
    mask_map_resized_arr = np.asarray(mask_map_resized)
    
    # Where qc_map_resized is black, set the corresponding pixel in tissue_map to black
    #tissue_map_arr[mask_map_resized_arr != mask_color] = 0
    tissue_map_arr[np.isin(mask_map_resized_arr, mask_color_list, invert=False)] = 0

    # Convert the result back to an image
    result_img = Image.fromarray(tissue_map_arr)
    return result_img

def downscale_patch(patch, downscale_factor):
    # Downscale patch coordinates with different rounding for start and end coordinates
    x1, y1, x2, y2 = patch
    downscaled_x1 = math.floor(x1 / downscale_factor)
    downscaled_y1 = math.floor(y1 / downscale_factor)
    downscaled_x2 = math.ceil(x2 / downscale_factor)
    downscaled_y2 = math.ceil(y2 / downscale_factor)
    return [downscaled_x1, downscaled_y1, downscaled_x2, downscaled_y2]

def create_heatmap_patches(patch_color, patch_size):
    # Create an image with the specified color and patch size
    heatmap_patch = Image.new("RGB", (patch_size, patch_size), color=patch_color)
    return heatmap_patch

def heatmap_color(input_list, heatmap_type):
    min_val = min(input_list)
    max_val = max(input_list)
    try:
        input_list = [(val - min_val) / (max_val - min_val) for val in input_list]
    except:
        input_list = input_list
    cmap = plt.get_cmap(heatmap_type)  # you can choose other colormaps like 'jet', 'viridis' etc.

    return [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in cmap(input_list)]

def reconstruct_image(mask_image, significant_indexes, patches, patches_downscaled):
    # Create a blank image of the same size as the mask_image
    reconstructed_image = Image.new('RGB', mask_image.size)

    # Iterate over the significant indexes
    for index, patch in zip(significant_indexes, patches):
        # Get the coordinates for the current patch from patches_downscaled
        x1, y1, x2, y2 = patches_downscaled[index]

        # Paste the patch into the reconstructed image at the calculated position
        patch = patch.resize((x2-x1, y2-y1))
        reconstructed_image.paste(patch, (x1, y1, x2, y2))

    return reconstructed_image

def normalize_cell_count_in_area(cell_count, area):
    normalized_count = []

    for i in range(len(cell_count)):
        try:
            normalized_count.append(cell_count[i]/area[i])
        except:
            normalized_count.append(0)
        
    return normalized_count

def calculate_ratio(cells_class, cells_area):
    percentage = []

    for i in range(len(cells_class)):
        try:
            percentage.append(cells_class[i]/cells_area[i])
        except:
            percentage.append(0)
        
    return percentage

def sum_of_lists(*lists):
    return [sum(values) for values in zip(*lists)]

def create_image_with_number(patch_size_mask, number, display_number_size=0.75):
    text_area = int(patch_size_mask * display_number_size)

    text_image = Image.new('RGB', (100, 9), color='black')
    draw_text_image = ImageDraw.Draw(text_image)

    # Add Text to an image
    draw_text_image.text((0, 0), str(number), fill=(255, 255, 255))

    width, height = text_image.size

    # Initialize the rightmost coordinate
    rightmost = 0

    # Iterate over all pixels to find the rightmost white pixel
    for x in range(width):
        for y in range(height):
            r, g, b = text_image.getpixel((x, y))
            if r == g == b == 255:  # This checks for white
                rightmost = max(rightmost, x)

    cropped_text_image = text_image.crop((0, 0, rightmost + 1, height))

    new_width = text_area

    original_width, original_height = cropped_text_image.size

    aspect_ratio = original_height / original_width
    new_height = int(aspect_ratio * new_width)

    resized_text_image = cropped_text_image.resize((new_width, new_height), 0)

    number_patch = Image.new('RGB', (patch_size_mask, patch_size_mask), color='black')

    # Calculate the position to paste, centered on the destination image
    paste_x = ((patch_size_mask//2) - (new_width//2))
    paste_y = ((patch_size_mask//2) - (new_height//2))

    # Create a copy of the destination image to preserve the original
    number_patch.paste(resized_text_image, (paste_x, paste_y))

    draw_patch = ImageDraw.Draw(number_patch)

    draw_patch.rectangle([(0, 0), (patch_size_mask-1, patch_size_mask-1)], outline="white", width=1)

    return number_patch

def apply_mask(image, mask):
    # Convert mask to grayscale (if it's not already) and then to '1' mode (binary)
    mask = mask.convert('L').point(lambda x: 0 if x<128 else 255, '1')

    # Apply mask to the image
    masked_image = Image.composite(image, Image.new('RGB', image.size), mask)

    return masked_image

def overlay_images(main_image, overlay_image, darkening_factor):
    # Resize overlay image to match the main image size
    overlay_resized = overlay_image.resize(main_image.size, 5)

    # Prepare the overlay image with the specified opacity
    overlay_with_alpha = overlay_resized.convert("RGBA")
    alpha = int(255 * darkening_factor)  # Convert darkening factor to alpha channel value
    overlay_with_alpha.putalpha(alpha)

    # Combine the images
    combined_image = Image.alpha_composite(main_image.convert("RGBA"), overlay_with_alpha)

    return combined_image.convert("RGB")

def normalize_coordinates(geometry, patch):
    min_x, min_y, _, _ = patch
    normalized = [(x - min_x, y - min_y) for x, y in geometry.exterior.coords]
    return normalized

def count_nuclei_in_geometry_collection(rtree_patches, geometry_collections):
    count_list = []
    for patch_id, (rtree, geometries) in enumerate(zip(rtree_patches, geometry_collections)):
        count = 0
        already_counted = set()  # To avoid counting duplicates

        for geom in geometries.geoms:
            # Convert to shapely geometry if not already one
            if not isinstance(geom, (Polygon, MultiPolygon)):
                print('had to convert')
                geom = shape(geom)

            # Find and count polygons in the R-tree that intersect with the geometry
            for idx in rtree.intersection(geom.bounds):
                if idx not in already_counted:
                    already_counted.add(idx)
                    count += 1

        count_list.append(count)
    return count_list

# Start processing each file

for filename in os.listdir(input_WSI_folder):
    if filename.endswith('.ndpi') or filename.endswith('.svs') or filename.endswith('.tiff'):
        print('Working on ' + filename)

        if output_exists(filename, output_path):
            print(f'Output already exists for {filename}, skipping')
            continue

        if filename.endswith('.ndpi'):
            fileend = '.ndpi'
        elif filename.endswith('.svs'):
            fileend = '.svs'
        elif filename.endswith('.tiff'):
            fileend = '.tiff'
        input_WSI = os.path.join(input_WSI_folder, filename)
        input_mask = os.path.join(input_mask_folder, f"{os.path.splitext(filename)[0]}{fileend}_mask_filter.png")
        input_geojson = os.path.join(input_geojson_folder, f"{os.path.splitext(filename)[0]}.geojson")
        input_qc = os.path.join(input_qc_folder, f"{os.path.splitext(filename)[0]}{fileend}_MASK.png")
        thumbnail = os.path.join(input_thumbnail_folder, f"{os.path.splitext(filename)[0]}{fileend}.jpg")
        include_image = os.path.join(include_image_folder, f"{os.path.splitext(filename)[0]}{fileend}.jpg")

        print('  opening files')

        # Load the .geojson file
        with open(input_geojson, 'r') as file:
            geojson = json.load(file)

        try:
            qc_image = Image.open(input_qc)
        except:
            print('  couldnt find qc')
            continue
        mask_image = Image.open(input_mask)
        thumbnail_image = Image.open(thumbnail)

        print('  loaded files')

        # Calculate how much the tissue map was downscaled by
        mask_downscale_factor, wsi_width, wsi_height, mpp = get_downscale_factor(input_WSI, original_mpp)
        print('  downscale factor is ' + str(mask_downscale_factor))

        # Fix mask aspect ratio by inserting it into a downscaled blank version of the WSI, so that we can overlay the mask on the thumbnail
        mask_image_empty = Image.new("L", (int(wsi_width / mask_downscale_factor), int(wsi_height / mask_downscale_factor)))
        mask_image_empty.paste(mask_image, (0, 0))
        mask_image = mask_image_empty

        # Generate a list of coordinates which correspond to each patch in the WSI
        patches = generate_patches(wsi_width, wsi_height, patch_size)
        number_of_patches = range(len(patches))

        # Downscale the coordinates for the mask image so that it gets the same patches as in the WSI
        patches_downscaled = [downscale_patch(patch, mask_downscale_factor) for patch in patches]
        patch_size_mask = patches_downscaled[0][2]

        print('  creating output files')
        output_path_json = output_path + 'csv/' + os.path.split(input_geojson)[1][:-8] + '_' + str(wsi_width) + '_' + str(wsi_height) + '_' + str(patch_size) + '_' + str(patch_size_mask) + '_patch_data.json'
        output_path_csv = output_path + 'csv/' + os.path.split(input_geojson)[1][:-8] + '_' + str(wsi_width) + '_' + str(wsi_height) + '_' + str(patch_size) + '_' + str(patch_size_mask) + '_patch_data.csv'
        output_path_heatmap = output_path + 'heatmaps/' + os.path.split(input_geojson)[1][:-8] + '_' + str(wsi_width) + '_' + str(wsi_height) + '_' + str(patch_size) + '_' + str(patch_size_mask) + '_'

        print('  splitting image into patches')

        # Split mask image into mask patches
        chunks_list_pre_qc = cut_image(mask_image, patches_downscaled)

        # Apply filtering to image, if qc or include are not enabled, just hand over the regular mask patches
        if use_qc:
            print('  applying quality control')

            # Calculate downscale factor for qc image, and generate equivalent patches
            qc_downscale_factor = wsi_width / qc_image.size[0]
            patches_downscaled_qc = [downscale_patch(patch, qc_downscale_factor) for patch in patches]

            # Split qc image into patches
            qc_chunks_list = cut_image(qc_image, patches_downscaled_qc)

            # Remove QC areas from mask patches
            chunks_list_pre_include = [exclude_only_areas(tissue_map, qc_map, qc_color) for tissue_map, qc_map in zip(chunks_list_pre_qc, qc_chunks_list)]
        else:
            chunks_list_pre_include = chunks_list_pre_qc

        if use_include:
            print('  applying include map')

            # Calculate downscale factor for include image, and generate equivalent patches
            include_downscale_factor = wsi_width / include_image.size[0]
            patches_downscaled_include = [downscale_patch(patch, include_downscale_factor) for patch in patches]

            # Split include image into patches
            include_chunks_list = cut_image(include_image, patches_downscaled_include)

            # Remove include areas from mask patches
            chunks_list = [include_only_areas(tissue_map, include_map, include_color) for tissue_map, include_map in zip(chunks_list_pre_include, include_chunks_list)]
        else:
            chunks_list = chunks_list_pre_include

        print('  turning maps into polygons')

        # Transform mask image patches into polygons
        poly_tumor_map_patches_small = [image_to_geometry_collection(chunk, tumor_color) for chunk in chunks_list]
        poly_stroma_map_patches_small = [image_to_geometry_collection(chunk, stroma_color) for chunk in chunks_list]

        # Upscale polygon map patches to the correct size, so it has the same resolution as the WSI 
        poly_tumor_map_patches_pre_significant = [scale_geometry_collection(geom_collection, mask_downscale_factor) for geom_collection in poly_tumor_map_patches_small]
        poly_stroma_map_patches_pre_significant = [scale_geometry_collection(geom_collection, mask_downscale_factor) for geom_collection in poly_stroma_map_patches_small]

        # Check the amount of tissue in each patch, and filter out any patches that are below min_mask_amount
        significant_indexes = get_significant_area_indices(poly_tumor_map_patches_pre_significant, poly_stroma_map_patches_pre_significant, patch_size, min_mask_amount)
        number_of_significant_indexes = range(len(significant_indexes))

        # Cut down patches to only include the indexes with a significant amount of tissue
        patches_significant = [patches[i] for i in significant_indexes] 

        # Remove any polygon map patches which are not significant
        poly_tumor_map_patches = [poly_tumor_map_patches_pre_significant[i] for i in significant_indexes]
        poly_stroma_map_patches = [poly_stroma_map_patches_pre_significant[i] for i in significant_indexes]

        print('  splitting nuclei into patches')

        # Get the features from our geojson file
        nuclei_data = geojson['features']

        # Define the nuclei classes
        nuclei_classes = ["Neutrophil", "Epithelial", "Lymphocyte", "Plasma", "Eosinophil", "Connective"]

        # Create an R-tree index for each class
        rtrees = {cls: rtree.index.Index() for cls in nuclei_classes}

        # Populate the R-trees with geometries
        for idx, feature in enumerate(nuclei_data):
            classification = feature['properties'].get('classification')
            if classification in rtrees:
                geom = shape(feature['geometry'])
                rtrees[classification].insert(idx, geom.bounds)

        # Initialize dictionaries for each nuclei type
        rtrees_patches_dict = {cls: [rtree.index.Index() for _ in patches] for cls in nuclei_classes}

        # Go through each patch, split out the nuclei classes, and make an rtree which contains them to add to a list, which are stored in a dictionary
        for patch_id, patch in enumerate(patches_significant):
            patch_box = box(*patch)
            for cls in nuclei_classes:
                for idx in rtrees[cls].intersection(patch):
                    geometry = shape(nuclei_data[idx]['geometry'])
                    if geometry.intersects(patch_box):
                        # Normalize coordinates relative to a patch
                        normalized_geom = normalize_coordinates(geometry, patch)
                        normalized_polygon = Polygon(normalized_geom)
                        rtrees_patches_dict[cls][patch_id].insert(idx, normalized_polygon.bounds)

        # Separate lists for each nuclei type, for ease of use
        rtrees_patches_neutrophil = rtrees_patches_dict["Neutrophil"]
        rtrees_patches_epithelial = rtrees_patches_dict["Epithelial"]
        rtrees_patches_lymphocyte = rtrees_patches_dict["Lymphocyte"]
        rtrees_patches_plasma = rtrees_patches_dict["Plasma"]
        rtrees_patches_eosinophil = rtrees_patches_dict["Eosinophil"]
        rtrees_patches_connective = rtrees_patches_dict["Connective"]

        # Create useful variables that contains the area of stroma, tumor, and both together (measured in square micrometers)
        area_patches_stroma = [sum(geom.area for geom in poly_stroma_map_patches[i].geoms if hasattr(geom, 'area')) * (mpp * mpp) for i in number_of_significant_indexes]
        area_patches_tumor = [sum(geom.area for geom in poly_tumor_map_patches[i].geoms if hasattr(geom, 'area')) * (mpp * mpp) for i in number_of_significant_indexes]
        area_patches_total = [a + b for a, b in zip(area_patches_tumor, area_patches_stroma)]

        # Calculate the number of cells in stroma areas (We're not doing epithelial because there shouldn't be any in the stroma)
        cells_in_stroma_epithelial = count_nuclei_in_geometry_collection(rtrees_patches_epithelial, poly_stroma_map_patches)

        # Calculate the amount of epithelial cells in tumor area, reclassifying the connective cells to epithelial ones, by simply counting how many connective cells are in tumor area, and adding that number to the epithelial count
        cells_in_tumor_epithelial_pre_correction = count_nuclei_in_geometry_collection(rtrees_patches_epithelial, poly_tumor_map_patches)
        cells_in_tumor_connective_to_reclassify = count_nuclei_in_geometry_collection(rtrees_patches_connective, poly_tumor_map_patches)
        cells_in_tumor_epithelial = [(cells_in_tumor_epithelial_pre_correction[entry] + cells_in_tumor_connective_to_reclassify[entry]) for entry in number_of_significant_indexes]
        cells_in_tumor_neutrophil = count_nuclei_in_geometry_collection(rtrees_patches_neutrophil, poly_tumor_map_patches)
        cells_in_tumor_lymphocyte = count_nuclei_in_geometry_collection(rtrees_patches_lymphocyte, poly_tumor_map_patches)
        cells_in_tumor_plasma = count_nuclei_in_geometry_collection(rtrees_patches_plasma, poly_tumor_map_patches)
        cells_in_tumor_eosinophil = count_nuclei_in_geometry_collection(rtrees_patches_eosinophil, poly_tumor_map_patches)
        cells_in_tumor_connective = count_nuclei_in_geometry_collection(rtrees_patches_connective, poly_tumor_map_patches)

        # Add all variables (which are are just a list of numbers for each patch to a single list, that we can export later)
        all_patch_variables = [
            'significant_indexes',
            'area_patches_stroma',
            'area_patches_tumor',
            'area_patches_total',
            'cells_in_tumor_epithelial_pre_correction',
            'cells_in_tumor_connective_to_reclassify',
            'cells_in_tumor_epithelial',
            'cells_in_tumor_neutrophil',
            'cells_in_tumor_lymphocyte',
            'cells_in_tumor_plasma',
            'cells_in_tumor_eosinophil',
            'cells_in_tumor_connective',
            'cells_in_stroma_epithelial',
        ]

        all_universal_variables = [
        ]

        # Export a CSV file with the per patch numbers

        print('  generating csv output')

        # Construct the dictionary using a dictionary comprehension and globals()
        patch_variables_dict = {var_name: globals()[var_name] for var_name in all_patch_variables}
        universal_variables_dict = {var_name: globals()[var_name] for var_name in all_universal_variables}
        combined_variable_dict = {**universal_variables_dict, **patch_variables_dict}

        # Create a DataFrame from the dictionary
        patch_variables_df = pd.DataFrame(patch_variables_dict)

        # Export to CSV, without the Pandas index, but with the header
        patch_variables_df.to_csv(output_path_csv, index=False)
