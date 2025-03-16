from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import math

Image.MAX_IMAGE_PIXELS = None

# Define locations for csv file and helper images
#output_folder = '/media/vmitchell/ssd8tb/01_CRC_UKK_Cut_NEW_2017_2019/'
output_folder = '/media/vmitchell/d9b4230a-e8fa-4b83-9965-fe2293d30b6a/TCGA_COADREAD_CSV/'

csv_folder = output_folder + 'csv/'
heatmap_folder = output_folder + 'heatmaps/'
heatmap_folder_blurred = output_folder + 'heatmaps/blurred/'
heatmap_folder_not_blurred = output_folder + 'heatmaps/not_blurred/'
heatmap_folder_blurred_global = output_folder + 'heatmaps/global_blurred/'
heatmap_folder_not_blurred_global = output_folder + 'heatmaps/global_not_blurred/'

os.makedirs(heatmap_folder_blurred, exist_ok=True)
os.makedirs(heatmap_folder_not_blurred, exist_ok=True)
os.makedirs(heatmap_folder_blurred_global, exist_ok=True)
os.makedirs(heatmap_folder_not_blurred_global, exist_ok=True)

# How much the thumbnail should be darkened in the heatmap images
darkening_factor = 0.5

# How much the patches should be blurred together, relative to the patch size
blur_number = 50

# Color map used for the heatmap
heatmap_type = 'rainbow' # good alternatives are 'jet' or 'turbo'

# Define functions

def generate_patches(wsi_width, wsi_height, patch_size):
    patches = []
    for y in range(0, wsi_height, patch_size):
        for x in range(0, wsi_width, patch_size):
            # Ensure xmax and ymax don't exceed the image dimensions
            xmax = min(x + patch_size, wsi_width)
            ymax = min(y + patch_size, wsi_height)
            
            patches.append([x, y, xmax, ymax])
    return patches

def downscale_patch(patch, downscale_factor):
    # Downscale patch coordinates with different rounding for start and end coordinates
    x1, y1, x2, y2 = patch
    downscaled_x1 = math.floor(x1 / downscale_factor)
    downscaled_y1 = math.floor(y1 / downscale_factor)
    downscaled_x2 = math.ceil(x2 / downscale_factor)
    downscaled_y2 = math.ceil(y2 / downscale_factor)
    return [downscaled_x1, downscaled_y1, downscaled_x2, downscaled_y2]

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

def heatmap_color_global(input_list, heatmap_type, global_min_max_value):
    min_val, max_val = global_min_max_value
    try:
        input_list = [(val - min_val) / (max_val - min_val) for val in input_list]
    except:
        input_list = input_list
    cmap = plt.get_cmap(heatmap_type)  # you can choose other colormaps like 'jet', 'viridis' etc.

    return [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in cmap(input_list)]

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


# Find the global min and max values for each variable, so we can universally normalize the heatmaps
global_max_min_values = {}

for file in os.listdir(csv_folder):
    if file.endswith(".csv") and file != 'image_level_data.csv':
        file_path = os.path.join(csv_folder, file)
        data = pd.read_csv(file_path)

        for column in data.columns:
            # Get min and max values from the current file
            min_val = data[column].min()
            max_val = data[column].max()

            # Update the dictionary with the min and max values
            if column not in global_max_min_values:
                global_max_min_values[column] = (min_val, max_val)
            else:
                current_min, current_max = global_max_min_values[column]
                global_max_min_values[column] = (min(current_min, min_val), max(current_max, max_val))

# Start processing csv files (skipping over the image_level_data file)

for file in os.listdir(csv_folder):
        if file.endswith(".csv") and file != 'image_level_data.csv':
            csv_file = csv_folder + file
            universal_mask_file = heatmap_folder + file[:-15] + '_universal_mask.png'
            thumbnail_file = heatmap_folder + file[:-15] + '_thumbnail.png'

            # Open helper images
            thumbnail_image = Image.open(thumbnail_file)
            mask_universal_image = Image.open(universal_mask_file)

            # Extract patch info from file name (using regex)
            patch_info = re.search(r"_(\d+)_(\d+)_(\d+)_(\d+)_patch_data", csv_file).groups()
            wsi_width, wsi_height, patch_size, patch_size_mask = int(patch_info[0]), int(patch_info[1]), int(patch_info[2]), int(patch_info[3])

            # Set output paths
            output_path_heatmap_not_blurred = heatmap_folder_not_blurred + file[:-15]
            output_path_heatmap_blurred = heatmap_folder_blurred + file[:-15]
            output_path_heatmap_not_blurred_global = heatmap_folder_not_blurred_global + file[:-15]
            output_path_heatmap_blurred_global = heatmap_folder_blurred_global + file[:-15]

            # Calculate blur factor and the factor by which the map image was downscaled
            blur_factor = patch_size // blur_number
            mask_downscale_factor = wsi_width/mask_universal_image.width

            # Read CSV file as a pandas dataframe
            csv_dataframe = pd.read_csv(csv_file)

            # Separate each column into an individual list
            variables_to_visualize = {col: csv_dataframe[col].tolist() for col in csv_dataframe.columns}

            # Seperate out the significant indexes (we need these to construct the heatmap image)
            significant_indexes = variables_to_visualize['significant_indexes']

            # Use the information from the file name to reconstruct the patches that were used in the image analysis
            patches = generate_patches(wsi_width, wsi_height, patch_size)
            patches_downscaled = [downscale_patch(patch, mask_downscale_factor) for patch in patches]

            # Start visualizing the numbers
            for var_name in variables_to_visualize:
                print('    generating heatmap output for variable ' + var_name)
                print('      creating heatmap images')

                # Take number for patch, normalize it and create a corresponding heatmap color
                try:
                    color_patches = heatmap_color(variables_to_visualize[var_name], heatmap_type)
                except:
                    print('      could not create heatmap color for variable ' + var_name)
                    continue
                # Do the same, but using the global max and min values
                color_patches_global = heatmap_color_global(variables_to_visualize[var_name], heatmap_type, global_max_min_values[var_name])

                print('      creating individual heatmap patches')

                # Create a small image patch with the color that was chosen
                heatmap_patches = [create_heatmap_patches(patch_color, patch_size_mask) for patch_color in color_patches]
                heatmap_patches_global = [create_heatmap_patches(patch_color, patch_size_mask) for patch_color in color_patches_global]

                print('      combining individual heatmap patches')

                # Combine the image patches of the colors to a complete image, of the same size as the mask
                heatmap_combined_image = reconstruct_image(mask_universal_image, significant_indexes, heatmap_patches, patches_downscaled)
                heatmap_combined_image_global = reconstruct_image(mask_universal_image, significant_indexes, heatmap_patches_global, patches_downscaled)

                print('      bluring heatmap image')

                # Blur the heatmap image
                blurred_heatmap_image = heatmap_combined_image.filter(ImageFilter.GaussianBlur(blur_factor))
                blurred_heatmap_image_global = heatmap_combined_image_global.filter(ImageFilter.GaussianBlur(blur_factor))

                print('      masking heatmap image')

                # Mask the heatmap images (the blurred version, and the not blurred version) with the tissue mask, to get a clean output
                masked_blurred_heatmap_image = apply_mask(blurred_heatmap_image, mask_universal_image)
                masked_heatmap_image = apply_mask(heatmap_combined_image, mask_universal_image)
                masked_blurred_heatmap_image_global = apply_mask(blurred_heatmap_image_global, mask_universal_image)
                masked_heatmap_image_global = apply_mask(heatmap_combined_image_global, mask_universal_image)

                print('      overlaying heatmap on thumbnail')

                # Overlay the heatmap images (the blurred version, and the not blurred version) on the thumbnail, applying darkening for visual clarity
                overlayed_blurred_heatmap_image = overlay_images(thumbnail_image, masked_blurred_heatmap_image, darkening_factor)
                overlayed_heatmap_image = overlay_images(thumbnail_image, masked_heatmap_image, darkening_factor)
                overlayed_blurred_heatmap_image_global = overlay_images(thumbnail_image, masked_blurred_heatmap_image_global, darkening_factor)
                overlayed_heatmap_image_global = overlay_images(thumbnail_image, masked_heatmap_image_global, darkening_factor)

                # Save outputs as images
                overlayed_blurred_heatmap_image.save(output_path_heatmap_blurred + '_' + var_name + '.png', 'PNG')
                overlayed_heatmap_image.save(output_path_heatmap_not_blurred + '_' + var_name + '_not_blurred' + '.png', 'PNG')
                overlayed_blurred_heatmap_image_global.save(output_path_heatmap_blurred_global + '_global_' + var_name + '.png', 'PNG')
                overlayed_heatmap_image_global.save(output_path_heatmap_not_blurred_global + '_global_' + var_name + '_not_blurred' + '.png', 'PNG')

                print('      finished generating heatmap output for variable ' + var_name)