import os
import numpy as np
from openslide import OpenSlide
from PIL import Image
import tensorflow as tf
from conic import predict
from stardist.models import StarDist2D
import rasterio.features
import json
from shapely.geometry import shape as shp
from scipy.ndimage import zoom

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
Image.MAX_IMAGE_PIXELS = None

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5240)])
  except RuntimeError as e:
    print(e)

# Input paths
input_folder = '/media/dr_pusher/YT_NVME_1/PROJECT_MELANOM_KA_WON/01_wsi_old/'
model_folder = 'model'
mask_folder = '/media/dr_pusher/YT_NVME_1/PROJECT_MELANOM_KA_WON/05_inference/02_output/mask/'

# Output path
output_folder = '/media/dr_pusher/YT_NVME_1/PROJECT_MELANOM_KA_WON/05_inference/output_cell/'

# Colors
tumor = 1
stroma = 2

magnification = 40

# Minimum amount of Tumor required for patch to be used
min_amount_of_tumor = 25

# Overlap to be used
mask_chunk_size = 64
overlap = 10

# How to split up the Files
divisions = 1
script_number = 1

# Optionally draw bounding boxes, for troubleshooting purposes
draw_bounding_boxes = False
# Optionally update the .geojson each chunk, for troubleshooting purposes
export_every_chunk = False
# Optionally draw masks, for troubleshooting purposes
show_masks = False

# Load TensorFlow model
model = StarDist2D(None, name='conic', basedir=model_folder)

# Define the class labels
class_labels = {
    0: "BACKGROUND",
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective"
}

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Determine Files to Process
number_of_files = num_files = len([f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))])
number_of_files_per_script = number_of_files / divisions
start = int(number_of_files_per_script * (script_number - 1))
end = int(number_of_files_per_script * script_number)
if script_number != 1:
    start += 1
elif script_number == divisions:
    end = number_of_files

# Function to shift nuclei shapes to global coordinates
def shift_coordinates(dictionary, x_offset, y_offset):
    if 'coordinates' in dictionary and isinstance(dictionary['coordinates'], list):
        for i, coordinates_set in enumerate(dictionary['coordinates']):
            if isinstance(coordinates_set, list):
                dictionary['coordinates'][i] = [(x + x_offset, y + y_offset) for x, y in coordinates_set]
    return dictionary

# Funtion to transform an array to a series of polygons
def create_geojson_from_array(array, mask_array,x_offset, y_offset, overlap, chunk_size):
    # Extract the instance map and classification channels
    instance_map = array[0].astype(np.int32)
    classification = array[1].astype(np.int32)

    # Create a list to store the GeoJSON features
    chunk_features = []
    masks = []

    # Generate mask polygon from tissue map
    mask = rasterio.features.shapes(mask_array)
    for shape, value in mask:
        if value == 1:
            global_pos_mask = shift_coordinates(shape, x_offset, y_offset)
            mask_shape = shp(global_pos_mask)
            masks.append(mask_shape)

            if show_masks == True:
                mask_chunk_feature = {"type": "Feature",
                        "geometry": global_pos_mask,
                        "properties": {"classification": "mask"}}
                
                # Add the feature to the list
                chunk_features.append(mask_chunk_feature)

    # Generate polygons from the instance map
    shapes = rasterio.features.shapes(instance_map)

    # Define the box that doesn't overlap
    bounding_box = {
                "type": "Polygon",
                "coordinates": [
                    ((overlap+x_offset, overlap+y_offset), (chunk_size - overlap+x_offset, overlap+y_offset), (chunk_size-overlap+x_offset, chunk_size-overlap+y_offset), (overlap+x_offset, chunk_size-overlap+y_offset), (overlap+x_offset, overlap+y_offset)),
                ]
            }

    # Convert box and mask to shape
    bounding_box_shape = shp(bounding_box)

    if draw_bounding_boxes:
        # Write the bounding box to the geojson file
        bounding_box_feat = {"type": "Feature",
                "geometry": bounding_box,
                "properties": {"classification": "Bounding Box"}}

        chunk_features.append(bounding_box_feat)

        # Write the chunk box to the geojson file
        chunk_box = {
                    "type": "Polygon",
                    "coordinates": [
                        ((x_offset, y_offset), (chunk_size+x_offset, y_offset), (chunk_size+x_offset, chunk_size+y_offset), (x_offset, chunk_size+y_offset), (x_offset, y_offset)),
                    ]
                }

        chunk_box_feat = {"type": "Feature",
                "geometry": chunk_box,
                "properties": {"classification": "Overlapping Box"}}

        chunk_features.append(chunk_box_feat)
    
    # Iterate over the shapes and create GeoJSON features
    for shape, value in shapes:
        if value != 0:  # Exclude the background
            shape = shift_coordinates(shape, x_offset, y_offset)

            # Get the centroid of the shape
            nucleus_shape = shp(shape)
            centroid = nucleus_shape.centroid

            # Only process shapes that are inside of the mask
            # Apply decision rule: only process shape if its centroid falls in non-overlapping region
            if centroid.within(bounding_box_shape):
                for part in masks:
                    if centroid.within(part):
                        # Get the classification label
                        class_label = class_labels[classification[np.where(instance_map == value)][0]]
                        
                        # Create the GeoJSON feature
                        chunk_feature = {"type": "Feature",
                                "geometry": shape,
                                "properties": {"classification": class_label}}
                        
                        # Add the feature to the list
                        chunk_features.append(chunk_feature)

    return chunk_features

print('processing files ' + str(start) + ' to ' + str(end))

# Iterate over WSI images
for filename in os.listdir(input_folder)[start:end]:
    print(filename)
    if filename.endswith('.ndpi') or filename.endswith('.svs') or filename.endswith('.tiff'):
        print('Working on ' + filename)

        print(output_folder + os.path.splitext(filename)[0] + '.geojson')
        if os.path.exists(output_folder + os.path.splitext(filename)[0] + '.geojson'):
            print('Output already exists, skipping file')
            continue

        wsi_path = os.path.join(input_folder, filename)
        #mask_path = os.path.join(mask_folder, f"{os.path.splitext(filename)[0]}.svs_mask_filter.png")
        mask_path = os.path.join(mask_folder, f"{os.path.splitext(filename)[0]}.ndpi_mask.png")
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_nuclei.png")

        # Load WSI and mask images
        try:
            mask = Image.open(mask_path).convert('RGB')
        except:
            print('matching mask does not exist, skipping file')
            continue

        wsi = OpenSlide(wsi_path)

        # Determi
        #mask_downscale_factor = wsi.dimensions[0] / mask.height
        try:
            mpp = float(wsi.properties["openslide.mpp-x"])
        except:
            mpp = 0.2325
        p_s = int(1 / mpp * 512)
        slide_width = wsi.level_dimensions[0][0]
        patch_numbers = slide_width / p_s
        proper_width = patch_numbers * 128
        mask_downscale_factor = slide_width / proper_width 

        wsi_chunk_size_non_rounded = mask_chunk_size * mask_downscale_factor
        wsi_chunk_size = int(wsi_chunk_size_non_rounded)

        # Overlap scaled up for WSI
        scaled_overlap = (overlap/2) * mask_downscale_factor

        print(mpp)
        print(p_s)
        print(slide_width)
        print(patch_numbers)
        print(mask_downscale_factor)
        print(wsi_chunk_size)
        print(mask_chunk_size)
        print(scaled_overlap)

        # Save the indexes of chunks with enough tumor or stroma pixels
        features = []

        # Iterate over chunks with overlap
        for y in range(0, mask.height - mask_chunk_size, mask_chunk_size - overlap):
            for x in range(0, mask.width - mask_chunk_size, mask_chunk_size - overlap):
                chunk = mask.crop((x, y, x + mask_chunk_size, y + mask_chunk_size))
                pixels = np.array(chunk)

                # Count tumor and stroma pixels
                tumor_pixels = np.sum(np.all(pixels == tumor, axis=2))
                stroma_pixels = np.sum(np.all(pixels == stroma, axis=2))

                # Check if the chunk has enough tumor or stroma pixels
                if tumor_pixels + stroma_pixels > min_amount_of_tumor:
                    print('processing patch [' + str(x) + ', ' + str(y) + ']')

                    index = (x * mask_downscale_factor, y * mask_downscale_factor)

                    # Create binary mask of the chunk
                    binary_mask = chunk.point(lambda x: 255 if x != 0 else 0).convert('1')

                    # Upscale the binary mask by the factor the mask was downscaled by
                    binary_mask = binary_mask.resize((int(binary_mask.width * mask_downscale_factor), int(binary_mask.height * mask_downscale_factor)), resample=Image.NEAREST)

                    # Process the WSI version of this chunk
                    chunk = wsi.read_region((int(index[0]), int(index[1])), 0, (wsi_chunk_size, wsi_chunk_size)).convert('RGB')
                    chunk_array = np.array(chunk)
                    print(chunk_array.shape)

                    if magnification == 40:
                        print('downscaling input')
                        chunk_array = zoom(chunk_array, (0.5, 0.5, 1), order=3)

                    output_chunk_array, count = predict(model, chunk_array,
                        normalize            = True,
                        test_time_augment    = False,
                        refine_shapes        = dict(),
                    )

                    if magnification == 40:
                        print('upscaling output')
                        output_chunk_array = zoom(output_chunk_array, (2, 2, 1), order=0)

                    # Convert mask to numpy array
                    mask_array = np.array(binary_mask).astype(np.int16)

                    # Fix array shape
                    output_chunk_array = np.stack([output_chunk_array[:, :, 0], output_chunk_array[:, :, 1]])

                    # Get the global coordinates for the chunk
                    x_offset = x * mask_downscale_factor
                    y_offset = y * mask_downscale_factor
                    
                    #print(x_offset, y_offset)

                    # Convert the inference map to geojson
                    geojson_data = create_geojson_from_array(output_chunk_array, mask_array, x_offset, y_offset, scaled_overlap, wsi_chunk_size_non_rounded)
                    if geojson_data:
                        features.extend(geojson_data)

                    # Export geojson every chunk
                    if export_every_chunk:
                        feature_collection = {"type": "FeatureCollection", "features": features}

                        # Write the GeoJSON data to a file
                        with open(output_folder + os.path.splitext(filename)[0] + '.geojson', "w") as f:
                            json.dump(feature_collection, f, indent = 3)

        # Create the GeoJSON feature collection
        feature_collection = {"type": "FeatureCollection", "features": features}

        # Write the GeoJSON data to a file
        with open(output_folder + os.path.splitext(filename)[0] + '.geojson', "w") as f:
            json.dump(feature_collection, f, indent = 3)

        # Close WSI image
        wsi.close()