import os
import json
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define base directory (assumes current script is in src/dataset)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))

# Define paths for each dataset split
sstubs_unique_dir = os.path.join(base_dir, "sstubs4j", "unique", "splits")
sstubs_repetition_dir = os.path.join(base_dir, "sstubs4j", "repetition", "splits")
defects4j_dir = os.path.join(base_dir, "defects4j", "splits")

# Define output directory (combined)
combined_dir = os.path.join(base_dir, "combined")
os.makedirs(combined_dir, exist_ok=True)

# A simple augmentation function (e.g., drop non-essential fields)
def augment_input(text):
    if "Project:" in text:
        text = text.split("Project:")[0].strip()
    return text

# Preprocess sstubs4j data (which contains both buggy and non-buggy examples)
def preprocess_sstubs(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    buggy_samples = []
    non_buggy_samples = []

    for bug in data:
        context_before = bug.get("contextBefore", "")
        context_after = bug.get("contextAfter", "")
        buggy_code = bug.get("buggyCode", "")
        source_after_fix = bug.get("sourceAfterFix", "")
        fix_commit_message = bug.get("fixCommitMessage", "")
        parent_commit_msg = bug.get("parentCommitMessage", "")
        bug_type = bug.get("bugType", "")

        # Construct texts for buggy (label 1) and non-buggy (label 0)
        buggy_text = (
            f"Context Before:\n{context_before}\n"
            f"Buggy Code: {buggy_code}\n"
            f"Commit Message: {fix_commit_message}\n"
            f"Parent Commit: {parent_commit_msg}\n"
            f"Bug Type: {bug_type}"
        )
        non_buggy_text = (
            f"Context After:\n{context_after}\n"
            f"Fixed Code: {source_after_fix}\n"
            f"Commit Message: {fix_commit_message}\n"
            f"Parent Commit: {parent_commit_msg}\n"
            f"Bug Type: {bug_type}"
        )

        buggy_text = augment_input(buggy_text)
        non_buggy_text = augment_input(non_buggy_text)

        buggy_samples.append(buggy_text)
        non_buggy_samples.append(non_buggy_text)

    logging.info(f"Loaded {len(buggy_samples)} buggy and {len(non_buggy_samples)} non-buggy examples from {file_path}")
    return buggy_samples, non_buggy_samples

# Preprocess defects4j data (which only has buggy examples)
def preprocess_defects4j(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    buggy_samples = []
    for bug in data:
        # Build a summary text for the bug
        failing_tests = bug.get("failingTests", [])
        tests = ", ".join([t.get("methodName", "") for t in failing_tests])
        buggy_text = (
            f"Code Diff:\n{bug.get('diff', '')}\n"
            f"Failing Tests: {tests}\n"
            f"Repair Patterns: {', '.join(bug.get('repairPatterns', []))}\n"
            f"Project: {bug.get('program', '')}"
        )
        buggy_text = augment_input(buggy_text)
        buggy_samples.append(buggy_text)
    logging.info(f"Loaded {len(buggy_samples)} buggy examples from {file_path}")
    return buggy_samples

# Function to combine a given split (train, val, or test)
def combine_split(split):
    # File names for each split
    sstubs_file = f"sstubsLarge-{split}.json"
    defects4j_file = f"defects4j-{split}.json"

    # Process sstubs4j (unique and repetition)
    unique_buggy, unique_non_buggy = preprocess_sstubs(os.path.join(sstubs_unique_dir, sstubs_file))
    repetition_buggy, repetition_non_buggy = preprocess_sstubs(os.path.join(sstubs_repetition_dir, sstubs_file))

    sstubs_buggy = unique_buggy + repetition_buggy
    sstubs_non_buggy = unique_non_buggy + repetition_non_buggy

    # Process defects4j (only buggy)
    defects4j_buggy = preprocess_defects4j(os.path.join(defects4j_dir, defects4j_file))

    # Combine buggy examples from both sources
    combined_buggy = sstubs_buggy + defects4j_buggy
    combined_non_buggy = sstubs_non_buggy  # Only from sstubs4j

    # Balance the classes by undersampling the majority class
    num_buggy = len(combined_buggy)
    num_non_buggy = len(combined_non_buggy)
    min_samples = min(num_buggy, num_non_buggy)

    balanced_buggy = random.sample(combined_buggy, min_samples)
    balanced_non_buggy = random.sample(combined_non_buggy, min_samples)

    logging.info(f"For split '{split}': {min_samples} buggy and {min_samples} non-buggy examples after balancing")

    # Create a list of dicts for the combined examples
    combined_data = []
    for text in balanced_buggy:
        combined_data.append({"text": text, "label": 1, "source": "buggy"})
    for text in balanced_non_buggy:
        combined_data.append({"text": text, "label": 0, "source": "non-buggy"})

    # Shuffle the data
    random.shuffle(combined_data)
    return combined_data

# Process each split and save to files
splits = ["train", "val", "test"]
for split in splits:
    logging.info(f"Processing split: {split}")
    combined_data = combine_split(split)
    output_file = os.path.join(combined_dir, f"combined-{split}.json")
    with open(output_file, "w") as f:
        json.dump(combined_data, f, indent=4)
    logging.info(f"Saved combined {split} dataset with {len(combined_data)} examples to {output_file}")

