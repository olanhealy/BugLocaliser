import os
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import sys

# ---------------------------
# LOGGING 
# ---------------------------
# Setup log directory in src, as this script takes time and runs in a tmux session
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "training_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Starting training script...")

# ---------------------------
# SET PATHS & ENVIRONMENT
# ---------------------------
# Set GPU to use CSIS server GPU 0. This was GPU with most free space
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define dataset file paths 
sstubs_train_path = "../../Data/sstubs4j/unique/splits/sstubsLarge-train.json"
sstubs_val_path   = "../../Data/sstubs4j/unique/splits/sstubsLarge-val.json"
sstubs_test_path  = "../../Data/sstubs4j/unique/splits/sstubsLarge-test.json"

defects_train_path = "../../Data/defects4j/splits/defects4j-train.json"
defects_val_path   = "../../Data/defects4j/splits/defects4j-val.json"
defects_test_path  = "../../Data/defects4j/splits/defects4j-test.json"

# Output directory for model
output_dir = "../../fine_tuned_model/"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# LOAD TOKENISER & CODEBERT
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# ---------------------------
# PREPROCESSING FUNCTION
# ---------------------------
def preprocess_data(data, dataset_type):
    """
    For sStuBs4J, each bug gives one buggy example (label 1) and one fixed example (label 0).
    For defects4j, we only have buggy examples (label 1), along with extra context.
    """
    inputs = []
    labels = []

    if dataset_type == "sstubs":
        for bug in tqdm(data, desc="Processing sStuBs4J data"):
            # Buggy example
            buggy_text = (
                f"Git Diff:\n{bug['fixPatch']}\n"
                f"Context Before:\n{bug['contextBefore']}\n"
                f"Buggy Code:\n{bug['buggyCode']}\n"
                f"Commit Message: {bug.get('fixCommitMessage', '')}"
            )
            inputs.append(buggy_text)
            labels.append(1)

            # Fixed example (negative example)
            fixed_text = (
                f"Git Diff:\n{bug['fixPatch']}\n"
                f"Context Before:\n{bug['contextBefore']}\n"
                f"Fixed Code:\n{bug['contextAfter']}\n"
                f"Commit Message: {bug.get('fixCommitMessage', '')}"
            )
            inputs.append(fixed_text)
            labels.append(0)

    elif dataset_type == "defects4j":
        for bug in tqdm(data, desc="Processing defects4j data"):
            buggy_text = (
                f"Git Diff:\n{bug['diff']}\n"
                f"Failing Tests:\n{json.dumps(bug.get('failingTests', []), indent=2)}\n"
                f"Repair Patterns: {', '.join(bug.get('repairPatterns', []))}"
            )
            inputs.append(buggy_text)
            labels.append(1)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    tokenized_data = tokenizer(inputs, truncation=True, padding=True, max_length=512)
    tokenized_data["labels"] = labels
    ds = Dataset.from_dict(tokenized_data)
    ds.set_format("torch") 
    return ds

# ---------------------------
# LOAD & PREPROCESS DATASETS
# ---------------------------
# Load sStubs4J datasets
with open(sstubs_train_path, "r") as f:
    sstubs_train_data = json.load(f)
with open(sstubs_val_path, "r") as f:
    sstubs_val_data = json.load(f)
with open(sstubs_test_path, "r") as f:
    sstubs_test_data = json.load(f)

sstubs_train = preprocess_data(sstubs_train_data, "sstubs")
sstubs_val   = preprocess_data(sstubs_val_data, "sstubs")
sstubs_test  = preprocess_data(sstubs_test_data, "sstubs")

# Load defects4j datasets
with open(defects_train_path, "r") as f:
    defects_train_data = json.load(f)
with open(defects_val_path, "r") as f:
    defects_val_data = json.load(f)
with open(defects_test_path, "r") as f:
    defects_test_data = json.load(f)

defects_train = preprocess_data(defects_train_data, "defects4j")
defects_val   = preprocess_data(defects_val_data, "defects4j")
defects_test  = preprocess_data(defects_test_data, "defects4j")

# ---------------------------
# PHASE 1: TRAIN ON SSTUBS4J
# ---------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "results_sstubs"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs_sstubs"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,         
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sstubs_train,
    eval_dataset=sstubs_val,
    tokenizer=tokenizer,
)

logging.info(">>> Phase 1: Training on sStuBs4J (balanced bug & fix examples)...")
trainer.train()

# Save intermediate sstuBs4J–finetuned model 
sstubs_model_dir = os.path.join(output_dir, "sstubs_model")
os.makedirs(sstubs_model_dir, exist_ok=True)
model.save_pretrained(sstubs_model_dir)
tokenizer.save_pretrained(sstubs_model_dir)
logging.info(f"Intermediate sStuBs4J model saved to {sstubs_model_dir}")

# ---------------------------
# PREPARE REPLAY BUFFER FROM sStubs4J
# ---------------------------
# Sampling 10% from sStuBs4J training set to "replay" during defects4j fine–tuning
random.seed(42)
replay_fraction = 0.1
replay_size = int(len(sstubs_train) * replay_fraction)
replay_indices = random.sample(range(len(sstubs_train)), replay_size)
sstubs_replay = sstubs_train.select(replay_indices)
logging.info(f"Replay buffer created with {replay_size} samples from sStuBs4J training set.")

# ---------------------------
# PHASE 2: FINE–TUNE ON defects4j WITH REPLAY
# ---------------------------
# We now fine–tune on defects4j (which only has buggy examples) but we mix in a small replay mini–batch from sStuBs4j
# This helps preserve the model’s ability to distinguish buggy vs. non–buggy code.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# Create DataLoaders for defects4j training and the replay buffer
defects_loader = DataLoader(defects_train, batch_size=12, shuffle=True)
replay_loader  = DataLoader(sstubs_replay, batch_size=4, shuffle=True)
replay_iter = iter(replay_loader)

# Use a lower learning rate to avoid overwriting previous knowledge
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs_defects = 3  # Set the number of epochs for defects4j training

logging.info(">>> Phase 2: Fine–tuning on defects4j with replay buffer...")
for epoch in range(num_epochs_defects):
    epoch_loss = 0.0
    num_batches = 0
    for defects_batch in tqdm(defects_loader, desc=f"Defects4j Epoch {epoch+1}/{num_epochs_defects}"):
        # Get a replay batch; if the replay loader is exhausted, restart it
        try:
            replay_batch = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_loader)
            replay_batch = next(replay_iter)

        # Move defects4j batch to device
        defects_batch = {k: v.to(device) for k, v in defects_batch.items()}
        outputs_defects = model(
            input_ids=defects_batch["input_ids"],
            attention_mask=defects_batch["attention_mask"],
            labels=defects_batch["labels"]
        )
        loss_defects = outputs_defects.loss

        # Move replay batch to device
        replay_batch = {k: v.to(device) for k, v in replay_batch.items()}
        outputs_replay = model(
            input_ids=replay_batch["input_ids"],
            attention_mask=replay_batch["attention_mask"],
            labels=replay_batch["labels"]
        )
        loss_replay = outputs_replay.loss

        # Combine the defects loss with the replay loss 
        total_loss = loss_defects + 0.5 * loss_replay

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += total_loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    logging.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

# Save the final model
final_model_dir = os.path.join(output_dir, "final_model")
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
logging.info(f"Final model saved to {final_model_dir}")

# ---------------------------
# EVALUATION
# ---------------------------
# Evaluation on sStuBs4J test set using the Huggingface Trainer
logging.info(">>> Evaluating on sStuBs4J test set...")
trainer.model = model  # update trainer's model if needed
sstubs_test_results = trainer.evaluate(sstubs_test)
logging.info(f"sStuBs4J Test Results: {sstubs_test_results}")

# Evaluation on defects4j test set using a custom loop
logging.info(">>> Evaluating on defects4j test set...")
defects_test_loader = DataLoader(defects_test, batch_size=16)
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in defects_test_loader:
        # Since the dataset is set to torch format, each field is a tensor
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(batch["labels"].cpu().numpy().tolist())

acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
logging.info(f"Defects4j Test Accuracy: {acc:.4f}")
logging.info(f"Defects4j Test Precision: {precision:.4f}")
logging.info(f"Defects4j Test Recall: {recall:.4f}")
logging.info(f"Defects4j Test F1 Score: {f1:.4f}")

logging.info("Training complete.")

