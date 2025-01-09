import json
import random
from pathlib import Path

# Paths
dataset_path = "../../Data/defects4j/defects4j-with-program.json"  # Input dataset (did the with program but doesnt really matter)
output_dir = "../../Data/defects4j/splits"  # Output directory

# Create the output directory 
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Load the dataset
with open(dataset_path, 'r') as f:
    data = json.load(f)

# Shuffle data for randomness
random.shuffle(data)

# Define splitting ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Calculate number of bugs for each split
total_bugs = len(data)
train_count = int(total_bugs * train_ratio)
val_count = int(total_bugs * val_ratio)
test_count = total_bugs - train_count - val_count  # Remaining bugs go to test

# Split the dataset
train_data = data[:train_count]
val_data = data[train_count:train_count + val_count]
test_data = data[train_count + val_count:]

# Save the splits to JSON files
def save_split(split_data, split_name):
    output_file = f"{output_dir}/{split_name}.json"
    with open(output_file, 'w') as f:
        json.dump(split_data, f, indent=4)
    print(f"Saved {split_name} split to {output_file}")

save_split(train_data, "defects4j-train")
save_split(val_data, "defects4j-val")
save_split(test_data, "defects4j-test")

