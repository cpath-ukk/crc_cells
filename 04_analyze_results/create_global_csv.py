import os
import pandas as pd
import re

csv_file_path = '/media/vmitchell/d9b4230a-e8fa-4b83-9965-fe2293d30b6a/TCGA_COADREAD_CSV/csv'  # Change this to the path of your zip file

def process_csv_files(csv_file_path, output_csv_path):
    # List all files in the subfolder
    csv_files = [f for f in os.listdir(csv_file_path) if f.endswith('.csv')]

    # Columns to sum
    columns_to_sum = [
        "area_patches_stroma", "area_patches_tumor", "area_patches_total",
        "cells_in_tumor_epithelial", "cells_in_tumor_lymphocyte", "cells_in_stroma_lymphocyte", "cells_in_stroma_plasma",
        "cells_in_stroma_neutrophil", "cells_in_stroma_eosinophil", "cells_in_stroma_connective",
        "cells_in_stroma_total"
    ]

    # Initialize a DataFrame to store the sums
    sums_df = pd.DataFrame(columns=["CSV_File"] + columns_to_sum)

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
            for column in columns_to_sum:
                sums[column] = 0  # Assign 0 to all variables
        else:
            # Sum the required columns
            for column in columns_to_sum:
                if column in df.columns:
                    sums[column] = df[column].sum()
                else:
                    sums[column] = None  # If the column does not exist in the file

        # Use concat instead of append
        sums_df = pd.concat([sums_df, pd.DataFrame([sums])], ignore_index=True)

    # Save the sums to a CSV file
    sums_df.to_csv(output_csv_path + '/image_level_data.csv', index=False)

    return sums_df

# Usage
process_csv_files(csv_file_path, csv_file_path)