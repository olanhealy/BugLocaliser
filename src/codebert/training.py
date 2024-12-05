import json
import os
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Set CUDA device to GPU 0 as free GPU available on server
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the pre-trained CodeBERT model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Set up the device for GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the base path for the datasets
data_base_path = '../../Data'

def load_dataset(base_path, category):
    """
    Load the train and test datasets for the whichever type its in *unique or repetition.

    Args:
        base_path (str): Base directory containing the data.
        category (str): Subdirectory for the specific dataset (e.g., 'unique' or 'repetition').

    Returns:
        tuple: Train and test datasets as lists of JSON objects.
    """

    train_path = os.path.join(base_path, category, 'sstubsLarge-train.json')
    val_path = os.path.join(base_path, category, 'sstubsLarge-val.json')

    # Load the train and validation data
    with open(train_path, 'r') as file:
        train_data = json.load(file)
    with open(val_path, 'r') as file:
        val_data = json.load(file)

    return train_data, val_data

# Function to label datasets
def label_data(data):
    """
    label the data for input to the model.

    Args:
        data (list): List of JSON entries.

    Returns:
        tuple: Encodings and labels for the dataset.
    """
    inputs = []
    labels = []
    for entry in tqdm(data, desc="Preprocessing Data"):
        # Add buggy code ("Before") as label 1
        inputs.append(f"Before: {entry['sourceBeforeFix']}")
        labels.append(1)
        # Add fixed code ("After") as label 0
        inputs.append(f"After: {entry['sourceAfterFix']}")
        labels.append(0)
    encodings = tokenizer(inputs, truncation=True, padding=True, max_length=512)
    return encodings, labels

# PyTorch dataset class
class BugDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch dataset for bug localisation.

    Args:
        encodings (dict): Tokenised inputs.
        labels (list): Corresponding labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Function to prepare data for a given category
def prepare_data(base_path, category):
    """
    Load and label train and validation data for the given category.

    Args:
        base_path (str): Base directory containing the data.
        category (str): Subdirectory for the specific dataset (e.g., 'unique' or 'repetition').

    Returns:
        tuple: Preprocessed train and validation datasets.
    """
    print(f"Preparing data for category: {category}")
    train_data, val_data = load_dataset(base_path, category)

    train_encodings, train_labels = label_data(train_data)
    val_encodings, val_labels = label_data(val_data)

    train_dataset = BugDataset(train_encodings, train_labels)
    val_dataset = BugDataset(val_encodings, val_labels)

    return train_dataset, val_dataset

# Load and prepare the unique dataset
unique_train, unique_val = prepare_data(data_base_path, 'unique')

# Training arguments for the unique dataset
unique_training_args = TrainingArguments(
    output_dir='./results_unique',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,  # Train for 3 epochs on the unique dataset
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    logging_dir='./logs_unique',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_num_workers=4,
)

# Train on the unique dataset
unique_trainer = Trainer(
    model=model,
    args=unique_training_args,
    train_dataset=unique_train,
    eval_dataset=unique_val,
)
print("Training on the 'unique' dataset...")
unique_trainer.train()

# Save the model after training on the 'unique' dataset
model.save_pretrained('./fine_tuned_codebert_unique')

# Load and prepare the repetition dataset
repetition_train, repetition_val = prepare_data(data_base_path, 'repetition')

# Training arguments for the repetition dataset
repetition_training_args = TrainingArguments(
    output_dir='./results_repetition',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,  # Fine-tune for 2 epochs on the repetition dataset
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    logging_dir='./logs_repetition',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_num_workers=4,
)

# Fine-tune on the repetition dataset
repetition_trainer = Trainer(
    model=model,
    args=repetition_training_args,
    train_dataset=repetition_train,
    eval_dataset=repetition_val,
)
print("Fine-tuning on the 'repetition' dataset...")
repetition_trainer.train()

# Save the final fine-tuned model
model.save_pretrained('./fine_tuned_codebert_final')
tokenizer.save_pretrained('./fine_tuned_codebert_final')

