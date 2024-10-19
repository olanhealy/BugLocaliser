import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# https://github.com/huggingface/transformers
# https://huggingface.co/microsoft/codebert-base

# Load pre-trained CodeBERT model and tokeniser
tokeniser = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Load Stubs$j dataset
with open('../Data/sstubsLarge-test.json', 'r') as file:
    dataset = json.load(file)

# first 10 bugs for training and the 11th for testing just small dont wanna kill me laptop
train_data = dataset[:10]
test_data = dataset[10]

# Preprocess function
def preprocess_data(data):
    inputs = []
    labels = []
    for entry in data:
        # label as 1 for buggy
        inputs.append(f"Before: {entry['sourceBeforeFix']}")
        labels.append(1)
        # Label as 2 for not buggy
        inputs.append(f"After: {entry['sourceAfterFix']}")
        labels.append(0)
    encodings = tokenizer(inputs, truncation=True, padding=True)
    return encodings, labels

# Tokenise and label the training data
train_encodings, train_labels = preprocess_data(train_data)

# Prepare training dataset
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

# Check if CUDA (GPU) is available, otherwise use CPU (maybe different when using csis labs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the correct device
model.to(device)

# Training arguments without evaluation
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    evaluation_strategy="no",  # Disable evaluation
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Test model on the 11th entry
test_inputs = tokenizer(f"Before: {test_data['sourceBeforeFix']}", return_tensors="pt", padding=True, truncation=True)
test_inputs = {key: val.to(device) for key, val in test_inputs.items()}  

# Perform inference
outputs = model(**test_inputs)
logits = outputs.logits
predicted_label = torch.argmax(logits, dim=1).item()

# Output the prediction result
if predicted_label == 1:
    print("Bug detected in the 11th code.")
else:
    print("No bug detected in the 11th code.")
