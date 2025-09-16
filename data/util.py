from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification

def preprocess_dataset(batch): 
    tags = ", ".join(batch["tags"])
    difficulty = batch["difficulty"]
    input_text = f"Generate a leetcode style coding problem with tags: {tags} and difficulty: {difficulty}. \n"
    target_text = batch["problem_description"]
        
    model_inputs = tokenizer(
        input_text, 
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    labels = tokenizer(
        target_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs