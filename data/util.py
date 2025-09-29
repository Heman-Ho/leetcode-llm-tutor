from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification

tokenizer=AutoTokenizer.from_pretrained("t5-small")

def preprocess_dataset(batch): 
    input_texts = []
    target_texts = []

    for tags_list, difficulty, problem_description in zip(batch["tags"], batch["difficulty"], batch["problem_description"]) :
        tags = ", ".join(tags_list)
        input_text = f"Generate a leetcode style coding problem with tags: {tags} and difficulty: {difficulty}. "
        input_texts.append(input_text)
        target_texts.append(problem_description)

    model_inputs = tokenizer(
        input_texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    labels = tokenizer(
        target_texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
