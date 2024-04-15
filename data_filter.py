import numpy as np
import pandas as pd
import os
import shutil

folder_path = 'aerial_60m_prunus_spec'
tol = 0.99

species = folder_path.split('_')[-2].capitalize() + '_' + folder_path.split('_')[-1]
print(species)
table = pd.read_csv('labels.csv')
table = table[table['File Name'].str.startswith(species)]

def extract_numbers(filename):
    return filename.split('_')[-3].replace('.tif', '')
table['File Number'] = table['File Name'].apply(extract_numbers)

table_numeric = table.apply(pd.to_numeric, errors='coerce')
columns_to_check = table_numeric.columns.difference(['File Number'])
filtered_table = table_numeric[(table_numeric[columns_to_check] >= tol).any(axis=1)]
filtered_numbers = filtered_table['File Number'].values.tolist()
print(filtered_numbers)
output_folder_path = 'prunus_spec'
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):
        # Extract the file number from the filename
        file_number = int(filename.split('_')[-3])  # Assuming file number is always before '_WEFL_NLF'

        # Check if the file number is in the list of numbers to keep
        if file_number in filtered_numbers:
            # Copy the image to the output folder
            shutil.copy(os.path.join(folder_path, filename), output_folder_path)
            print(f"File {filename} copied to the output folder.")

