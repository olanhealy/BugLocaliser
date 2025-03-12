import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ---------------------------
# LOAD MODEL AND TOKENIZER
# ---------------------------
# Update this path to point to your fine-tuned model directory.
model_dir = "../../../fine_tuned_model/model_final"
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# PREPROCESSING FUNCTION
# ---------------------------
def preprocess_ide_input(context, code_change, commit_message, parent_commit_message, bug_type, project_name, is_buggy=True):
    """
    Format the IDE input to match the training format.
    For buggy examples, use 'Context Before:' and 'Buggy Code:'
    For fixed/non-buggy examples, use 'Context After:' and 'Fixed Code:'
    """
    if is_buggy:
        formatted_input = (
            f"Context Before:\n{context}\n"
            f"Buggy Code: {code_change}\n"
            f"Commit Message: {commit_message}\n"
            f"Parent Commit: {parent_commit_message}\n"
            f"Bug Type: {bug_type}\n"
            f"Project: {project_name}"
        )
    else:
        formatted_input = (
            f"Context After:\n{context}\n"
            f"Fixed Code: {code_change}\n"
            f"Commit Message: {commit_message}\n"
            f"Parent Commit: {parent_commit_message}\n"
            f"Bug Type: {bug_type}\n"
            f"Project: {project_name}"
        )
    return formatted_input

# ---------------------------
# CLASSIFICATION FUNCTION
# ---------------------------
def classify_code_change(input_text):
    # Tokenise the input text to match the model's training format.
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
    
    return pred, confidence

# ---------------------------
# MAIN DEMO SCRIPT
# ---------------------------
def main():
    # Mock input from an IDE for a buggy change.
    context_buggy = (
        "public void update(double deltaTime) {\n"
        "    if (mPaused)\n"
        "        return;\n"
        "    // Additional code...\n"
        "}"
    )
    code_change_buggy = "if (mPaused)"  # A simplified snippet of the change.
    commit_message_buggy = "Fixing bug in update method"
    parent_commit_message_buggy = "Initial commit message"
    bug_type_buggy = "LESS_SPECIFIC_IF"
    project_name_buggy = "MyJavaProject"
    
    # Create input string mimicking the training format for a buggy example.
    input_text_buggy = preprocess_ide_input(
        context=context_buggy,
        code_change=code_change_buggy,
        commit_message=commit_message_buggy,
        parent_commit_message=parent_commit_message_buggy,
        bug_type=bug_type_buggy,
        project_name=project_name_buggy,
        is_buggy=True
    )
    
    # Mock input from an IDE for a non-buggy (fixed) change.
    context_not_buggy = (
        "public void update(double deltaTime) {\n"
        "    if (mPaused || !mPlaying)\n"
        "        return;\n"
        "    // Additional code...\n"
        "}"
    )
    code_change_not_buggy = "if (mPaused || !mPlaying)"  # The corrected code snippet.
    commit_message_not_buggy = "Refactoring update method for clarity"
    parent_commit_message_not_buggy = "Initial commit message"
    bug_type_not_buggy = "LESS_SPECIFIC_IF"
    project_name_not_buggy = "MyJavaProject"
    
    input_text_not_buggy = preprocess_ide_input(
        context=context_not_buggy,
        code_change=code_change_not_buggy,
        commit_message=commit_message_not_buggy,
        parent_commit_message=parent_commit_message_not_buggy,
        bug_type=bug_type_not_buggy,
        project_name=project_name_not_buggy,
        is_buggy=False
    )
    
    # Classify both examples.
    pred_buggy, conf_buggy = classify_code_change(input_text_buggy)
    pred_not_buggy, conf_not_buggy = classify_code_change(input_text_not_buggy)
    
    label_map = {0: "Not Buggy", 1: "Buggy"}
    print("=== IDE Demo Classification ===\n")
    print("Buggy Example:")
    print(input_text_buggy)
    print(f"\nPrediction: {label_map[pred_buggy]} (Confidence: {conf_buggy:.4f})\n")
    
    print("Not Buggy Example:")
    print(input_text_not_buggy)
    print(f"\nPrediction: {label_map[pred_not_buggy]} (Confidence: {conf_not_buggy:.4f})\n")

if __name__ == "__main__":
    main()

