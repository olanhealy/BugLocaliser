import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained CodeBERT model and tokeniser
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Load Stubs4J datasets
with open('../Data/repitition/sstubsLarge-train.json', 'r') as file:
    train_data = json.load(file)

with open('../Data/repitition/sstubsLarge-test.json', 'r') as file:
    test_data = json.load(file)

# Preprocess function
def preprocess_data(data):
    inputs = []
    labels = []
    for entry in data:
        # Label as 1 for buggy
        inputs.append(f"Before: {entry['sourceBeforeFix']}")
        labels.append(1)
        # Label as 0 for fixed
        inputs.append(f"After: {entry['sourceAfterFix']}")
        labels.append(0)
    encodings = tokenizer(inputs, truncation=True, padding=True, max_length=512)
    return encodings, labels

# Preprocess the datasets
train_encodings, train_labels = preprocess_data(train_data)
test_encodings, test_labels = preprocess_data(test_data)

# Prepare datasets for PyTorch
class BugDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BugDataset(train_encodings, train_labels)
test_dataset = BugDataset(test_encodings, test_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,  
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    logging_dir='./logs',
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save model checkpoints
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the model and tokeniser
model.save_pretrained('./fine_tuned_codebert')
tokenizer.save_pretrained('./fine_tuned_codebert')

# Test the model on a few samples from the test dataset
for i in range(5):  # Test on the first 5 samples
    test_input = tokenizer(
        f"Before: {test_data[i]['sourceBeforeFix']}",
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    test_input = {key: val for key, val in test_input.items()}
    outputs = model(**test_input)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    if predicted_label == 1:
        print(f"Bug detected in sample {i + 1}.")
    else:
        print(f"No bug detected in sample {i + 1}.")

