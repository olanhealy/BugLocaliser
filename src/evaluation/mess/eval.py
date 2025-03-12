import torch
import os
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ---------------------------
# LOAD MODEL AND TOKENIZER
# ---------------------------
model_dir = "../../../fine_tuned_model/model_final"
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# TEST SAMPLES (Modify these)
# ---------------------------
test_samples = [
    {
        "description": "Buggy Code: Incorrect function name",
        "code": "int result = caluclateSum(a, b);"
    },
    {
        "description": "Buggy Code: Missing return statement",
        "code": "public int add(int a, int b) { a + b; }"
    },
    {
        "description": "Correct Code: Proper function call",
        "code": "int result = calculateSum(a, b);"
    },
    {
        "description": "Correct Code: Includes return statement",
        "code": "public int add(int a, int b) { return a + b; }"
    }
]

# ---------------------------
# PROCESS & CLASSIFY
# ---------------------------
def classify_code(samples):
    for sample in samples:
        # Tokenize input
        inputs = tokenizer(sample["code"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Run model inference
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)  # Convert logits to probabilities
            pred = torch.argmax(probs, dim=-1).item()
            probability = probs[0][pred].item()  # Get probability of chosen class

        # Print prediction
        label = "Buggy" if pred == 1 else "Not Buggy"
        print(f"[{sample['description']}] → Prediction: **{label}** (Confidence: {probability:.4f})")

# Run classification
classify_code(test_samples)

