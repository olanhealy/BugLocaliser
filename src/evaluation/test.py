import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model (update the model_dir if needed)
model_dir = os.path.abspath(os.path.join("..", "..", "fine_tuned_model", "model_final"))
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.to(device)
model.eval()

def predict_buggy(code_snippet):
    """
    Predict whether the given code snippet is buggy (label 1) or not buggy (label 0).
    """
    # You may want to add a prompt or structure similar to your training examples.
    # For instance, if your training examples start with "Context Before:" for buggy code,
    # you can optionally add a prefix to mimic the training format.
    input_text = code_snippet.strip()
    
    # Tokenise the input text
    encoded = tokenizer(input_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    encoded = {key: val.to(device) for key, val in encoded.items()}
    
    # Run inference
    with torch.no_grad():
        logits = model(**encoded).logits
        prediction = torch.argmax(logits, dim=-1).item()
    
    # Interpret prediction
    return "Buggy" if prediction == 1 else "Not Buggy"

# Example: using a randomly made-up code snippet.
# You can replace this with any code string or even a function that generates random code.
sample_code = """
Context Before:
public void doSomething() {
    // Some random code
    int x = 10;
    if (x > 5) {
        System.out.println("x is greater than 5");
    }
    // Buggy Code: missing closing bracket
Commit Message: Dummy commit message for testing
Parent Commit: Dummy parent commit
Bug Type: SAMPLE_BUG_TYPE
"""

result = predict_buggy(sample_code)
print("Prediction for the sample code:", result)

