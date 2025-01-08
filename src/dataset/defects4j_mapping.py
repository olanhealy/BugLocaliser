import os
import json
from tqdm import tqdm

# Paths for dataset and output
DATASET_PATH = "../../Data/defects4j/defects4j-preprocessed.json"
OUTPUT_PATH = "../../Data/defects4j/defects4j-with-program.json"

# Mapping keywords in file paths to program names to represent github names
FILE_KEYWORDS_TO_PROGRAM = {
    # JFreeChart
    "org/jfree": "jfreechart",
    # Commons Libraries
    "org/apache/commons/lang3": "commons-lang",
    "org/apache/commons/lang": "commons-lang",
    "org/apache/commons/math3": "commons-math",
    "org/apache/commons/math": "commons-math",
    "org/apache/commons/cli": "commons-cli",
    "org/apache/commons/codec": "commons-codec",
    "org/apache/commons/collections": "commons-collections",
    "org/apache/commons/compress": "commons-compress",
    "org/apache/commons/csv": "commons-csv",
    "org/apache/commons/jxpath": "commons-jxpath",
    # Closure Compiler
    "google/javascript": "closure-compiler",
    # Gson
    "com/google/gson": "gson",
    # Jackson Core
    "jackson/core": "jackson-core",
    # Jackson Databind
    "jackson/databind": "jackson-databind",
    # Jackson XML
    "jackson/dataformat/xml": "jackson-dataformat-xml",
    # Jsoup
    "org/jsoup": "jsoup",
    # Mockito
    "mockito": "mockito",
    # Joda-Time
    "org/joda": "joda-time",
}

# Infer program from changedFiles paths
def infer_program(changed_files):
    # Direct matching based on keywords
    for file in changed_files:
        for keyword, program in FILE_KEYWORDS_TO_PROGRAM.items():
            if keyword in file:
                return program

    # No match found
    return None

# Process dataset to add program field
def add_program_to_dataset():
    with open(DATASET_PATH, "r") as file:
        dataset = json.load(file)

    updated_dataset = []
    missing_programs = []

    for bug in tqdm(dataset, desc="Processing Bugs"):
        changed_files = bug.get("changedFiles", [])
        program = infer_program(changed_files)

        if not program:
            # Log unmapped bugs to command output
            missing_programs.append({
                "bugId": bug["bugId"],
                "changedFiles": changed_files
            })
            bug["program"] = "NO DATA AVAILABLE"
        else:
            bug["program"] = program

        updated_dataset.append(bug)

    # Save updated dataset
    with open(OUTPUT_PATH, "w") as outfile:
        json.dump(updated_dataset, outfile, indent=4)
    print(f"Updated dataset with programs saved to {OUTPUT_PATH}")

    # Log missing programs
    if missing_programs:
        print("\nBugs with missing program mapping:")
        for item in missing_programs:
            print(f"  - Bug ID: {item['bugId']}, Changed Files: {item['changedFiles']}")

if __name__ == "__main__":
    add_program_to_dataset()

