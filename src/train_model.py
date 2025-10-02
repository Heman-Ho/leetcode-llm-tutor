from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration
from data import util
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
checkpoint = "t5-small"
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