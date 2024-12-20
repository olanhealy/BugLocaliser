import os
import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

# Set up device for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path to model directory
model_dir = os.path.abspath('../models/fine_tuned_codebert_final')

# Load the fine-tuned CodeBERT model and tokeniser
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = RobertaForSequenceClassification.from_pretrained(model_dir)
model.to(device)

# Load test datasets
with open('../../Data/sstubs4j/unique/sstubsLarge-test.json', 'r') as file:
    unique_data = json.load(file)

with open('../../Data/sstubs4j/repetition/sstubsLarge-test.json', 'r') as file:
    repetition_data = json.load(file)

# Combine unique and repetition test datasets
test_data = unique_data + repetition_data

# Preprocess test data to be like training 
def preprocess_data(data):
    inputs = []
    labels = []
    for entry in data:
        inputs.append(f"Before: {entry['sourceBeforeFix']}")
        labels.append(1)
        inputs.append(f"After: {entry['sourceAfterFix']}")
        labels.append(0)
    encodings = tokenizer(inputs, truncation=True, padding=True, max_length=512, return_tensors="pt")
    return encodings, labels

# Preprocess the combined test dataset
test_encodings, test_labels = preprocess_data(test_data)

# Move encodings and labels to the correct device
test_encodings = {key: val.to(device) for key, val in test_encodings.items()}
test_labels = torch.tensor(test_labels).to(device)

# Batch size for evaluation
batch_size = 16

# Predictions and labels lists
all_predictions = []
all_labels = []

# Evaluation loop with batching and progress tracking
print("Starting evaluation...")
model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(test_labels), batch_size)):
        batch_encodings = {key: val[i:i+batch_size] for key, val in test_encodings.items()}
        outputs = model(**batch_encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(test_labels[i:i+batch_size].cpu().numpy())

        # Print progress every 10 batches
        if (i // batch_size) % 10 == 0:
            print(f"Processed {(i + batch_size)} samples out of {len(test_labels)}")

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Display metrics
print("\nEvaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

