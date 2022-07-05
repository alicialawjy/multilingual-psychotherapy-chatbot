from transformers import AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoModelWithLMHead, pipeline

def load_dataset(train_path, tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, data_collator

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/mGPT")
model = AutoModelWithLMHead.from_pretrained("sberbank-ai/mGPT")
train_path = 'data/epzh-for-gpt.txt'
train_dataset, data_collator = load_dataset(train_path, tokenizer)

training_args = TrainingArguments(
    output_dir="rewriting/gpt2-ep", #The output directory
    overwrite_output_dir=True,      #overwrite the content of the output directory
    num_train_epochs=3,             # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    # eval_steps = 400,             # Number of update steps between two evaluations.
    save_steps=800,                 # after # steps model is saved 
    warmup_steps=500,               # number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()

# save model to output_dir
trainer.save_model()

# test the model
ep_generator = pipeline('text-generation',model='rewriting/gpt2-ep', tokenizer='sberbank-ai/mGPT',config={'max_length':50})
ep_generator('悲伤 - 这是由最近或遥远的事件（或多个事件）引起的吗?')[0]['generated_text']