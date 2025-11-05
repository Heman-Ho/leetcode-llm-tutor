from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration
from data import util
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
checkpoint = "Salesforce/codet5-small"
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the dataset
dataset_train = load_dataset("newfacade/LeetCodeDataset", split="train")
dataset_test = load_dataset("newfacade/LeetCodeDataset", split="test")

# apply tokenization in batches 
print("Tokenizing the dataset...")
tokenized_train = dataset_train.map(util.preprocess_dataset, batched=True)
tokenized_test = dataset_test.map(util.preprocess_dataset, batched=True)



# Set the format of the dataset to be compatible with PyTorch
print("setting the format...")
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("training the model ...")
# Set up the training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir = "./checkpoints",
    eval_strategy = "epoch",
    learning_rate = 5e-5, 
    weight_decay = 0.01,
    num_train_epochs = 3,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    predict_with_generate = True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model, 
    args=training_args,
    train_dataset=tokenized_train, 
    eval_dataset=tokenized_test,
    tokenizer=tokenizer, 
    data_collator=data_collator
)

trainer.train()

# Example of inference after traininig: 
def generate_problem(tags: list[str], difficulty: str) -> str: 
    input_text = f"Generate a leetcode style coding problem with tags: {', '.join(tags)} and difficulty: {difficulty}."
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generating a sample problem with tags ['array', 'hash table'] and difficulty 'Medium':")
print(generate_problem(["array", "hash table"], "Medium"))