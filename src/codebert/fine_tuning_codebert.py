import json
import os
import torch
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from tqdm import tqdm

# Set GPU to 0 to utilise most free GPU on ul server as at 20-01-2025
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define paths for datasets relative path
sstubs_train_path = "../../Data/sstubs4j/unique/splits/sstubsLarge-train.json"
sstubs_test_path = "../../Data/sstubs4j/unique/splits/sstubsLarge-test.json"
sstubs_val_path = "../../Data/sstubs4j/unique/splits/sstubsLarge-val.json"

defects_train_path = "../../Data/defects4j/splits/defects4j-train.json"
defects_test_path = "../../Data/defects4j/splits/defects4j-test.json"
defects_val_path = "../../Data/defects4j/splits/defects4j-val.json"

# Define output directory
output_dir = "../../fine_tuned_model/"
os.makedirs(output_dir, exist_ok=True)

# Load CodeBERT tokeniser
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Preprocessing function to only take in needed data
def preprocess_data(data, dataset_type):
    inputs = []
    labels = []

    for bug in tqdm(data, desc=f"Processing {dataset_type} data"):
        if dataset_type == "sstubs":
            buggy_text = (
                f"Git Diff:\n{bug['fixPatch']}\n"
                f"Context Before:\n{bug['contextBefore']}\n"
                f"Buggy Code:\n{bug['buggyCode']}\n"
                f"Commit Message: {bug.get('fixCommitMessage', '')}"
            )
            inputs.append(buggy_text)
            labels.append(1)

            fixed_text = (
                f"Git Diff:\n{bug['fixPatch']}\n"
                f"Context Before:\n{bug['contextBefore']}\n"
                f"Fixed Code:\n{bug['contextAfter']}\n"
                f"Commit Message: {bug.get('fixCommitMessage', '')}"
            )
            inputs.append(fixed_text)
            labels.append(0)

        elif dataset_type == "defects4j":
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

    return Dataset.from_dict(tokenized_data)

# Load datasets
sstubs_train = preprocess_data(json.load(open(sstubs_train_path)), "sstubs")
sstubs_val = preprocess_data(json.load(open(sstubs_val_path)), "sstubs")
sstubs_test = preprocess_data(json.load(open(sstubs_test_path)), "sstubs")

defects_train = preprocess_data(json.load(open(defects_train_path)), "defects4j")
defects_val = preprocess_data(json.load(open(defects_val_path)), "defects4j")
defects_test = preprocess_data(json.load(open(defects_test_path)), "defects4j")

# Load the CodeBERT model
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # Train on each dataset for 3 epochs
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Train on sstubs dataset first
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=sstubs_train,
    eval_dataset=sstubs_val,
    tokenizer=tokenizer,
)
print("Training on sstubs dataset...")
trainer.train()

# Save intermediate model
sstubs_model_dir = os.path.join(output_dir, "sstubs_model")
os.makedirs(sstubs_model_dir, exist_ok=True)
model.save_pretrained(sstubs_model_dir)
tokenizer.save_pretrained(sstubs_model_dir)

# Continue training on defects4j dataset
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=defects_train,
    eval_dataset=defects_val,
    tokenizer=tokenizer,
)
print("Fine-tuning on defects4j dataset...")
trainer.train()

# Save final model
final_model_dir = os.path.join(output_dir, "final_model")
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# Evaluate on test sets
print("Evaluating on sstubs test set...")
sstubs_test_results = trainer.evaluate(sstubs_test)
print("sStuBs4J Test Results:", sstubs_test_results)

print("Evaluating on defects4j test set...")
trainer.eval_dataset = defects_test
defects_test_results = trainer.evaluate()
print("Defects4J Test Results:", defects_test_results)

