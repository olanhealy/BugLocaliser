import json
import pandas as pd

# Load defects4j raw dataset
file_path = "../../Data/defects4j/defects4j-raw.json"
with open(file_path, "r") as f:
    data = json.load(f)

# storage for structured data
records = []

# Process each bug in file by id
for bug in data:
    bug_id = bug.get("bugId", "")
    changed_files = list(bug.get("changedFiles", {}).keys())
    diff = bug.get("diff", "")
    failing_tests = [
        {
            "className": test.get("className", ""),
            "methodName": test.get("methodName", ""),
            "error": test.get("error", ""),
            "message": test.get("message", "")
        }
        for test in bug.get("failingTests", [])
    ]
    repair_actions = bug.get("repairActions", [])
    repair_patterns = bug.get("repairPatterns", [])
    metrics = bug.get("metrics", {})

    # Add structured record
    records.append({
        "bugId": bug_id,
        "changedFiles": changed_files,
        "diff": diff,
        "failingTests": failing_tests,
        "repairActions": repair_actions,
        "repairPatterns": repair_patterns,
        "metrics": metrics
    })

# Save structured data for further use
output_json = "../../Data/defects4j/defects4j-preprocessed.json"
with open(output_json, "w") as f:
    json.dump(records, f, indent=4)

print(f"Preprocessed data saved to {output_json}")


