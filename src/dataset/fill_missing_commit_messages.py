import os
import json

# Paths to the dataset folders
BASE_DIR = "../../Data/sstubs4j"
DATASETS = ["unique/enhanced", "repetition/enhanced"]
FILES = ["sstubsLarge-train.json", "sstubsLarge-test.json", "sstubsLarge-val.json"]

def fill_missing_commit_messages(folder_path, file_path):
    """
    Fill missing commit messages in the dataset with 'NO DATA AVAILABLE' as some git commands couldnt work, repos may be private etc
    """
    dataset_path = os.path.join(folder_path, file_path)
    print(f"Processing file: {dataset_path}")
    try:
        with open(dataset_path, "r") as file:
            data = json.load(file)

        for bug in data:
            if not bug.get("fixCommitMessage"):
                bug["fixCommitMessage"] = "NO DATA AVAILABLE"
            if not bug.get("parentCommitMessage"):
                bug["parentCommitMessage"] = "NO DATA AVAILABLE"

        # Save the updated dataset
        with open(dataset_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Updated file saved: {dataset_path}")
    except Exception as e:
        print(f"Error processing {dataset_path}: {e}")

# Process each dataset folder
for dataset_folder in DATASETS:
    folder_path = os.path.join(BASE_DIR, dataset_folder)
    for dataset_file in FILES:
        dataset_path = os.path.join(folder_path, dataset_file)
        if os.path.exists(dataset_path):
            fill_missing_commit_messages(folder_path, dataset_file)
        else:
            print(f"File not found: {dataset_path}")

