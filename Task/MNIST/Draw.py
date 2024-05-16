import matplotlib.pyplot as plt
import pandas as pd
import os

# Directory containing the CSV files
directory = '/mnt/data'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Sort the files to ensure they are in the correct order
csv_files.sort()

# Create subplots
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 24))
axes = axes.flatten()

# Iterate through the CSV files and plot the cost functions
for i, csv_file in enumerate(csv_files):
    # Read the CSV file
    file_path = os.path.join(directory, csv_file)
    df = pd.read_csv(file_path, header=None)

    # Extract the cost values
    cost_values = df[1]

    # Plot the cost values
    axes[i].plot(cost_values)
    axes[i].set_title(csv_file)
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Cost')

# Adjust layout
plt.tight_layout()
plt.show()
