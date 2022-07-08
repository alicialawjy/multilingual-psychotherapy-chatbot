from transformers import AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, EarlyStoppingCallback, \
    Trainer, TrainingArguments, AutoModelWithLMHead, GPT2LMHeadModel, TextGenerationPipeline
from torch.utils.data import TensorDataset, Dataset
import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import time

def load_dataset(train_path, tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=100)
     
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, data_collator

############# Data Loader for GPT-2 ############# 
def encoded_df(df, supervised, tokenizer):
    # extract df columns
    gender = df['gender'].values.tolist()
    age = df['age'].values.tolist()
    emotion = df['emotion'].values.tolist()
    base = df['base'].values.tolist()
    rewriting = df['rewriting'].values.tolist()

    # concatenate df columns horizontally, joining with the respective tokens
    formatted_input = []
    for (g, a, e, b, r) in list(zip(gender, age, emotion, base, rewriting)):
        input = '[PROMPT]' + g + '[SEP]' + a + '[SEP]' + e + '[SEP]' + b + '[REWRITE]'
        # if supervised, append the rewritings as well
        if supervised:
            input += r
        
        formatted_input.append(input)

    # encode the formatted input
    encoded_input = tokenizer(formatted_input, 
                            return_tensors = 'pt',
                            padding = True,
                            truncation = True
                            )

    return encoded_input

    # return {'gender': gender, 'age': age, 'emotion': emotion, 'base': base, 'rewriting': rewriting}

class GPT2RewritingDataset(Dataset):
    ''' 
    DataLoader for GPT-2 Rewriting Task 

    '''
    def __init__(self, tokenizer, encodings, supervised=True): # ok
        self.encodings = encodings
        self.tokenizer = tokenizer
        self.supervised = supervised

    def __len__(self): # ok
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx): 
        # # need to pass items to collate_fn with the following acceptable keys:
        # # ['input_ids', 'past_key_values', 'attention_mask', 'token_type_ids', 'position_ids', 'head_mask', 
        # # 'inputs_embeds','encoder_hidden_states', 'encoder_attention_mask', 'labels', 'use_cache', 
        # # 'output_attentions', 'output_hidden_states','return_dict', 'labels', 'label', 'label_ids']
        # formatted = '[PROMPT]' + self.input['gender'][idx] + '[SEP]' + self.input['age'][idx] + '[SEP]' + self.input['emotion'][idx] + \
        # '[SEP]' + self.input['base'][idx] + '[REWRITE]'
        return {'labels': self.encodings['input_ids'][idx], 'input_ids': self.encodings['input_ids'][idx], 'attention_mask': self.encodings['attention_mask'][idx], 'token_type_ids': self.encodings['token_type_ids'][idx]}
        # return {'return_dict': {'gender': self.input['gender'][idx], 'age':self.input['age'][idx], 'emotion': self.input['emotion'][idx], 'base': self.input['base'][idx], 'rewriting': self.input['rewriting'][idx]}}
 
    # Takes batches of the input data
    def collate_fn(self, batch):
        # format into tensors
        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        for b in batch:
            input_ids.append(list(b['input_ids']))
            attention_mask.append(list(b['attention_mask']))
            token_type_ids.append(list(b['token_type_ids']))
            labels.append(list(b['labels']))
        
        return {'labels': torch.tensor(labels), 'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask), 'token_type_ids': torch.tensor(token_type_ids)}

############# GPT-2 Custom Trainer ############# 
# class GPT2_Trainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # inputs: tokenized encoding(dict_form)
    #     # As followed from https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel.forward.example
    #     print(inputs)
    #     outputs = model(**inputs)
    #     loss = outputs.loss         # next sentence prediction loss
    #     # logits = outputs.logits   # (prediction score for each word)

    #     return loss


############# Main Code ############# 
if __name__ == "__main__":
    output_dir = 'rewriting/gpt2-supervised'
    os.environ["WANDB_DISABLED"] = "true"

    # Fix Device
    GPU = True
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using {device}")

    ##### G P T - 2 #####
    # Model
    PRE_TRAINED_MODEL_NAME = 'uer/gpt2-chinese-cluecorpussmall'
    model = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)              # uses the same tokenizer as the original gpt-2
    additional_tokens = {'rewrite_token':'[REWRITE]', 'prompt_token':'[PROMPT]'}    # additional tokens for conditional generation
    tokenizer.add_tokens(list(additional_tokens.values()), special_tokens=True)           # add into tokenizer vocabulary
    for token_name, token in additional_tokens.items():
        setattr(tokenizer, token_name, token)                                       # assign corr. names (used in dataloader)

    model.resize_token_embeddings(len(tokenizer))                                   # resize the model token embedding space
    tokenizer.save_pretrained(output_dir)                                           # save the new tokenizer in the model directory

    ##### D A T A S E T S #####
    # for Part 1 of the Pipeline - generic EmpatheticPersonas
    # DataFrames
    df_generic = pd.read_csv('data/empathy/EP_empathy_2144_ZH.csv', index_col=0).head(15)
    df_generic_train, df_generic_test = train_test_split(df_generic, test_size=0.2, shuffle=True, random_state=0) 
    df_generic_val, df_generic_test = train_test_split(df_generic_test, test_size=0.5, shuffle=True, random_state=0)

    # Format and encode df with encoded_df()
    dict_generic_train = encoded_df(df=df_generic_train, supervised=True, tokenizer=tokenizer) 
    dict_generic_val = encoded_df(df=df_generic_val, supervised=False, tokenizer=tokenizer) 
    dict_generic_test = encoded_df(df=df_generic_test, supervised=False, tokenizer=tokenizer) 

    # Get DataLoader object, used by Trainer
    # (set supervised = False for validation and test sets: i.e. don't append rewritings)
    generic_train_dataset = GPT2RewritingDataset(tokenizer=tokenizer, encodings=dict_generic_train, supervised=True) 
    generic_val_dataset = GPT2RewritingDataset(tokenizer=tokenizer, encodings=dict_generic_val, supervised=False)  
    generic_test_dataset = GPT2RewritingDataset(tokenizer=tokenizer, encodings=dict_generic_test, supervised=False) 

    ##### T R A I N I N G #####
    # Early Stopping Module
    trainer_callback = EarlyStoppingCallback(early_stopping_patience = 5,
                                            early_stopping_threshold = 0.001)

    # Training Arguments
    training_args = TrainingArguments(output_dir = output_dir,              # Output directory where checkpoints + models are saved
                                    overwrite_output_dir = True,            # Overwrite the output directory if populated
                                    learning_rate = 1e-5,                   # Learning rate
                                    num_train_epochs = 1,                   # Number of training epochs
                                    warmup_steps = 50,
                                    per_device_train_batch_size = 4,        # Batch size for training
                                    # Early Stopping Arguments
                                    per_device_eval_batch_size = 4,         # Batch size for evaluation
                                    evaluation_strategy = 'steps',          # Number of update steps between two evaluations
                                    eval_steps = 2,                         # Evaluate every 50 steps
                                    save_strategy = 'steps',                # Save strategy
                                    save_steps = 2,                        # Save every 50 steps
                                    save_total_limit = 2,                   # Save only the 5 latest models. Deletes older models
                                    logging_strategy = 'steps',             # Logging strategy
                                    logging_dir = 'rewriting/gpt2-supervised/logs',
                                    logging_steps = 2,                     # Log every 100 steps
                                    include_inputs_for_metrics = True,
                                    metric_for_best_model = 'eval_loss',    # Decide based on eval_loss
                                    greater_is_better = False,              # Lower eval_loss is better
                                    load_best_model_at_end = True           # Required by EarlyStoppingCallback
                                    )

    # Trainer
    trainer = Trainer(args=training_args,
                    model = model,
                    tokenizer = tokenizer,
                    train_dataset = generic_train_dataset,
                    eval_dataset = generic_val_dataset,
                    data_collator = generic_train_dataset.collate_fn,
                    # compute_metrics = compute_metrics,              # needed by Trainer.evaluate
                    callbacks = [trainer_callback]                  # EarlyStoppingCallback module
                    )

    trainer.train()

    # Test the model
    # model = AutoModelWithLMHead.from_pretrained("rewriting/gpt2-supervised") # use our trained 
    # tokenizer = AutoTokenizer.from_pretrained("rewriting/gpt2-supervised") # uses the same tokenizer as the original gpt-2

    base_utterance = '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]这是由特别事件引起的吗？,我很遗憾听到你感觉不舒服。有什么特别的事情让你有这种感觉吗？[REWRITE]'
    input_ids = tokenizer.encode(base_utterance, return_tensors = 'pt')
    output = model.generate(input_ids, 
                            max_length = 100, 
                            num_return_sequences= 3,
                            num_beams = 5,
                            no_repeat_ngram_size = 2,
                            clean_up_tokenization_spaces=True,
                            return_full_text=False,
                            clean_up_tokenization_spaces=True,
                            early_stopping = True)

    rewritings = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
    print(rewritings)

