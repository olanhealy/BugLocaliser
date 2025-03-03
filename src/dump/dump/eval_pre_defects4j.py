import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_file = "evaluation_pre_defects4j_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("Starting evaluation of pre-defects4j models...")

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokeniser
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Model directories
model_dirs = {
    "model_unique": "../../fine_tuned_model/model_unique",
    "model_repetition": "../../fine_tuned_model/model_repetition"
}

# Test set paths
unique_test_path = os.path.join("..", "..", "Data", "sstubs4j", "unique", "splits", "sstubsLarge-test.json")
repetition_test_path = os.path.join("..", "..", "Data", "sstubs4j", "repetition", "splits", "sstubsLarge-test.json")
defects4j_test_path = os.path.join("..", "..", "Data", "defects4j", "splits", "defects4j-test.json")

# function to preprocess test data
def preprocess_test_data(file_path, dataset_type="sstubs4j"):
    with open(file_path, "r") as f:
        test_data = json.load(f)
    inputs = []
    labels = []
    for bug in tqdm(test_data, desc=f"Processing {file_path}"):
        context_before = bug.get("contextBefore", "")
        context_after = bug.get("contextAfter", "")
        fix_commit_message = bug.get("fixCommitMessage", "")
        parent_commit_msg = bug.get("parentCommitMessage", "")
        bug_type = bug.get("bugType", "")
        project_name = bug.get("projectName", "")

        if dataset_type == "sstubs4j":
            buggy_text = f"Context Before:\n{context_before}\nBug Type: {bug_type}\nProject: {project_name}"
            inputs.append(buggy_text)
            labels.append(1)
            not_buggy_text = f"Context After:\n{context_after}\nBug Type: {bug_type}\nProject: {project_name}"
            inputs.append(not_buggy_text)
            labels.append(0)
        else:
            buggy_text = f"Context Before:\n{context_before}\nBug Type: {bug_type}\nProject: {project_name}"
            inputs.append(buggy_text)
            labels.append(1)
    return inputs, labels

# Evaluate the models
for model_name, model_dir in model_dirs.items():
    logging.info(f"Evaluating {model_name}...")
    model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    model.to(device)
    model.eval()
    
    for test_name, test_path in zip(["UNIQUE", "REPETITION", "DEFECTS4J"], [unique_test_path, repetition_test_path, defects4j_test_path]):
        dataset_type = "sstubs4j" if "sstubs4j" in test_path else "defects4j"
        inputs, labels = preprocess_test_data(test_path, dataset_type=dataset_type)
        encoded_inputs = tokenizer(inputs, truncation=True, padding=True, max_length=512, return_tensors='pt')
        encoded_inputs = {key: val.to(device) for key, val in encoded_inputs.items()}
        labels = torch.tensor(labels).to(device)

        with torch.no_grad():
            logits = model(**encoded_inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        labels = labels.cpu().numpy()
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        report = classification_report(labels, preds, target_names=["Not Buggy", "Buggy"], labels=[0, 1], digits=4)
        cm = confusion_matrix(labels, preds, labels=[0, 1])

        logging.info(f"{model_name} on {test_name} Test Set Results:")
        logging.info(f"Accuracy: {acc:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")
        logging.info(f"Classification Report:\n{report}")
        logging.info(f"Confusion Matrix:\n{cm}")

logging.info("Evaluation of pre-defects4j models complete.")

