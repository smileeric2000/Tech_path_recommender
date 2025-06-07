from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset


#Load pretrained GPT-2
model_name  =  'gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

#Pad tokens
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))


#Load dataset for model fine-tuning
def format_training(example):
    prompt = example["prompt"].strip()
    response = example["response"].strip()
    return {"text": f"{prompt} {response}"}

dataset = load_dataset("json", data_files={"train": "gpt2_course_prompts.jsonl"})
dataset = dataset["train"].map(format_training)


#Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['text'], padding = 'max_length', truncation = True, max_length = 600)

tokenized_dataset = dataset.map(tokenize, batched = True)

#Define data collector <<masking>>
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

#Define Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt_path",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=False,  # Set to True if you have a GPU
    report_to="none"
)



#Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset
)

trainer.train()