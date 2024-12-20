import json
from collections import Counter

# Load the preprocessed dataset
file_path = "../../Data/defects4j/defects4j-preprocessed.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Calculate the number of bugs
num_bugs = len(data)

# Calculate the average number of changed files per bug
changed_files_counts = [len(bug.get("changedFiles", [])) for bug in data]
avg_changed_files = sum(changed_files_counts) / num_bugs if num_bugs > 0 else 0

# Calculate the distribution of repair actions
repair_actions_list = [action for bug in data for action in bug.get("repairActions", [])]
repair_actions_distribution = Counter(repair_actions_list)

# Print the results
print("Statistics for Defects4J Dataset:")
print(f"Number of Bugs: {num_bugs}")
print(f"Average Changed Files per Bug: {avg_changed_files:.2f}")
print("Distribution of Repair Actions:")
for action, count in repair_actions_distribution.items():
    print(f"  {action}: {count}")

# Save stats to a JSON file
output_path = "../../Data/defects4j/defects4j-stats.json"
stats = {
    "num_bugs": num_bugs,
    "avg_changed_files": avg_changed_files,
    "repair_actions_distribution": repair_actions_distribution
}

# Convert Counter to a regular dictionary
stats["repair_actions_distribution"] = dict(repair_actions_distribution)

with open(output_path, "w") as f:
    json.dump(stats, f, indent=4)

print(f"Statistics saved to {output_path}")

