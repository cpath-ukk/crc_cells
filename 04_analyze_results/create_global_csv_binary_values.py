import os
import pandas as pd
import re

csv_file_path = '/Volumes/SSD/Uni/PhD/BINARY MASKS/HALLE_CSV/csv'  # Change this to the path of your zip file

cutoff_values = {
    'normalized_cells_in_stroma_lymphocyte': 0.000920,
    'normalized_cells_in_tumor_lymphocyte': 0.000660395,
    'normalized_cells_in_stroma_eosinophil': 0.000226299,
    'normalized_cells_in_stroma_neutrophil': 0.0000699,
    'percentage_cells_in_stroma_connective': 0.6565298
}

cutoff_keys = list(cutoff_values.keys())

def process_csv_files(csv_file_path, output_csv_path):
    # List all files in the subfolder
    csv_files = [f for f in os.listdir(csv_file_path) if f.endswith('.csv')]

    # Initialize a DataFrame to store the sums
    sums_df = pd.DataFrame(columns=["CSV_File"] + cutoff_keys)

    # Process each valid CSV file
    for csv_file in csv_files:
        # Read the CSV file
        file_path = os.path.join(csv_file_path, csv_file)
        df = pd.read_csv(file_path)

        try:
            file_name = re.search(r'^(.*?)_\d+_\d+_\d+_\d+_patch_data$', os.path.splitext(csv_file)[0]).group(1)
        except:
            file_name = csv_file

        # Initialize a dictionary to store sums for this file
        sums = {"CSV_File": file_name}

        if df.empty:  # If the CSV file has headers but no rows
            for column in  cutoff_keys:
                sums[column] = 0  # Assign 0 to all variables
        else:
            for column in cutoff_keys: 
                if column in df.columns:
                    below_cutoff = df[column] < cutoff_values[column]
                    above_cutoff = df[column] >= cutoff_values[column]
                    
                    #print(f'Below cutoff: {below_cutoff.sum()}')
                    #print(f'Above cutoff: {above_cutoff.sum()}')

                    # Calculate the percentage of rows above the cutoff
                    if above_cutoff.sum() == 0:
                        percentage_above_cutoff = 0
                    elif below_cutoff.sum() == 0:
                        percentage_above_cutoff = 1
                    else:
                        percentage_above_cutoff = above_cutoff.sum() / (above_cutoff.sum() + below_cutoff.sum())

                    #print(f'Percentage above cutoff: {percentage_above_cutoff}')
                    
                    # Save the percentage to a column named like the dictionary key
                    sums[column] = percentage_above_cutoff
                else:
                    sums[column] = None  # If the column does not exist in the file

        # Use concat instead of append
        sums_df = pd.concat([sums_df, pd.DataFrame([sums])], ignore_index=True)

    # Save the sums to a CSV file
    sums_df.to_csv(output_csv_path + '/image_level_data_binary_percentage_patches_above_cutoff.csv', index=False)

    return sums_df

# Usage
process_csv_files(csv_file_path, csv_file_path)