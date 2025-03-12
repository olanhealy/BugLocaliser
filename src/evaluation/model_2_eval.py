import os
import json
import torch
import logging
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_file = "evaluation_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Starting evaluation script with combined test dataset...")

# ---------------------------
# SET DEVICE CONFIGURATION
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# DEFINE PATHS
# ---------------------------
# Assuming this script is run from within src/ (or adjust accordingly)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))
combined_dir = os.path.join(base_dir, "combined")
test_file = os.path.join(combined_dir, "combined-test.json")

# ---------------------------
# LOAD TOKENIZER & MODEL
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Load the fine-tuned model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fine_tuned_model", "model_final"))
logging.info(f"Loading model from: {model_dir}")
model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.to(device)
model.eval()

# ---------------------------
# DATA LOADING & TOKENISATION
# ---------------------------
def load_and_tokenize(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    # Create a Hugging Face Dataset from the list
    dataset = Dataset.from_list(data)
    # Tokenise the 'text' field
    tokenized = tokenizer(dataset["text"], truncation=True, padding=True, max_length=512)
    tokenized["labels"] = dataset["label"]
    tokenized_dataset = Dataset.from_dict(tokenized)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

logging.info("Loading and tokenising test dataset...")
test_dataset = load_and_tokenize(test_file)

# ---------------------------
# EVALUATION
# ---------------------------
logging.info("Starting model inference on test dataset...")

# Run inference
all_preds = []
all_labels = []
batch_size = 8
for i in tqdm(range(0, len(test_dataset), batch_size)):
    batch = test_dataset[i:i+batch_size]
    # Move inputs to device
    inputs = {key: val.to(device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}
    labels = batch["labels"].to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)
    
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Compute metrics
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
report = classification_report(all_labels, all_preds, target_names=["Not Buggy", "Buggy"], digits=4)
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

logging.info("Test Set Evaluation Results:")
logging.info(f"Accuracy: {acc:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Classification Report:\n{report}")
logging.info(f"Confusion Matrix:\n{cm}")

# Save confusion matrix plot if needed
if len(set(all_labels)) > 1:
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Buggy", "Buggy"], yticklabels=["Not Buggy", "Buggy"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for Combined Test Set")
    plt.savefig("combined_test_confusion_matrix.png")
    plt.close()
    
logging.info("Evaluation complete.")

