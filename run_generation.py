from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import TensorDataset, Dataset
import os
import torch
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt
import random

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

    return formatted_input, encoded_input

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
        return {'labels': self.encodings['input_ids'][idx], 'input_ids': self.encodings['input_ids'][idx], 'attention_mask': self.encodings['attention_mask'][idx], 'token_type_ids': self.encodings['token_type_ids'][idx]}
 
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
def run_supervised():
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
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)               # uses the same tokenizer as the original gpt-2
    additional_tokens = {'rewrite_token':'[REWRITE]', 'prompt_token':'[PROMPT]'}    # additional tokens for conditional generation
    tokenizer.add_tokens(list(additional_tokens.values()), special_tokens=True)     # add into tokenizer vocabulary
    for token_name, token in additional_tokens.items():
        setattr(tokenizer, token_name, token)                                       # assign corr. names (used in dataloader)

    model.resize_token_embeddings(len(tokenizer))                                   # resize the model token embedding space
    tokenizer.save_pretrained(output_dir)                                           # save the new tokenizer in the model directory

    ##### D A T A S E T S #####
    # for Part 1 of the Pipeline - generic EmpatheticPersonas
    # DataFrames
    df_generic = pd.read_csv('data/empathy/EP_empathy_2144_ZH.csv', index_col=0)
    df_generic_train, df_generic_test = train_test_split(df_generic, test_size=0.2, shuffle=True, random_state=0) 
    df_generic_val, df_generic_test = train_test_split(df_generic_test, test_size=0.5, shuffle=True, random_state=0)

    # Format and encode df with encoded_df()
    _, dict_generic_train = encoded_df(df=df_generic_train, supervised=True, tokenizer=tokenizer) 
    _, dict_generic_val = encoded_df(df=df_generic_val, supervised=False, tokenizer=tokenizer) 
    _, dict_generic_test = encoded_df(df=df_generic_test, supervised=False, tokenizer=tokenizer) 

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
                                    num_train_epochs = 50,                  # Number of training epochs
                                    warmup_steps = 50,
                                    per_device_train_batch_size = 4,        # Batch size for training
                                    # Early Stopping Arguments
                                    per_device_eval_batch_size = 4,         # Batch size for evaluation
                                    evaluation_strategy = 'steps',          # Number of update steps between two evaluations
                                    eval_steps = 50,                        # Evaluate every 50 steps
                                    save_strategy = 'steps',                # Save strategy
                                    save_steps = 50,                        # Save every 50 steps
                                    save_total_limit = 5,                   # Save only the 5 latest models. Deletes older models
                                    logging_strategy = 'steps',             # Logging strategy
                                    logging_dir = 'rewriting/gpt2-supervised/logs',
                                    logging_steps = 50,                     # Log every 100 steps
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
                    # compute_metrics = compute_metrics,                # needed by Trainer.evaluate
                    callbacks = [trainer_callback]                      # EarlyStoppingCallback module
                    )

    trainer.train()

    trainer.save_model(output_dir='rewriting/gpt2-supervised/best-model')

    # Test the model
    # model = AutoModelWithLMHead.from_pretrained("rewriting/gpt2-supervised") # use our trained 
    # tokenizer = AutoTokenizer.from_pretrained("rewriting/gpt2-supervised") # uses the same tokenizer as the original gpt-2

    prompt = '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]这是由特别事件引起的吗？[REWRITE]'
    input_ids = tokenizer.encode(prompt, return_tensors = 'pt').to(device)
    output = model.generate(input_ids, 
                            max_length = 100, 
                            do_sample=True, 
                            temperature=1.5,
                            top_k=50, 
                            top_p=0.95, 
                            num_return_sequences= 3,
                            num_beams = 5,
                            # no_repeat_ngram_size = 2,
                            clean_up_tokenization_spaces=True,
                            return_full_text=False,
                            early_stopping = True)

    rewritings = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

    for i, r in enumerate(rewritings):
        print(f"{i}: {r}")


def run_RL():
    ##### P A R A M E T E R S ######
    config = {
        "lm_name": "rewriting/gpt2-supervised/best-model",                  # generative model (gpt2)
        "empathy_classifier_name": "empathy_classifier/binary-empathy",     # empathy classifier (xlm-r)
        "steps": 51200,                                                     # aka epochs = steps/batch_size = 51200/256 = 200 epochs
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "max_len": 100,
        "lr": 1e-5,
        "init_kl_coef":0.2,
        "seed": 1,
        #"target": 6,
        #"horizon":10000,
        #"gamma":1,
        #"lam":0.95,
        #"cliprange": .2,
        #"cliprange_value":.2,
        #"vf_coef":.1, 
    }

    # random seed
    np.random.seed(config['seed'])

    # Set up W&B logger
    wandb.init(project='satbot', config=config)

    ##### Fix Device ######
    GPU = True
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using {device}")

    ##### M O D E L S  &  T O K E N I S E R S #####
    # Empathy Classifier
    EMPATHY_CLASSIFIER_NAME = config['empathy_classifier_name']
    empathy_classifier = AutoModelForSequenceClassification.from_pretrained(EMPATHY_CLASSIFIER_NAME).to(device)
    empathy_tokenizer = AutoTokenizer.from_pretrained(EMPATHY_CLASSIFIER_NAME)

    # GPT2 Language Models
    # NOTE: need to change line8 in gpt2.py of trl lib from transformers.modeling_utils to generation_utils
    GPT2_PRETRAINED_NAME = config['lm_name']
    gpt2_model = GPT2HeadWithValueModel.from_pretrained(GPT2_PRETRAINED_NAME).to(device)        # model to be finetuned
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(GPT2_PRETRAINED_NAME).to(device)    # reference model
    gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_PRETRAINED_NAME)                        # gpt2 tokenizer

    wandb.watch(gpt2_model, log='all')

    ##### L O A D  D A T A S E T S #####
    df = pd.read_csv('data/empathy/trl_train.csv', index_col=0) # DataFrame
    dict_train_text, dict_train_encoded = encoded_df(df=df, supervised=False, tokenizer=gpt2_tokenizer) # format and encode
    train_dataloader = GPT2RewritingDataset(tokenizer=gpt2_tokenizer, encodings=dict_train_encoded, supervised=True) # dataloader object
    
    ##### P P O  R L  T R A I N I N G  L O O P #####
    ppo_trainer = PPOTrainer(model=gpt2_model, ref_model=gpt2_model_ref, tokenizer=gpt2_tokenizer, **config)
    fbs = config['forward_batch_size']  # forward batch size

    for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()
        
        # Batch prompts
        batch_idx = random.choices(range(train_dataloader.__len__()),k=config['batch_size'])
        batch_dict_list = [train_dataloader.__getitem__(n) for n in batch_idx]
        batch_dict = train_dataloader.collate_fn(batch_dict_list)['input_ids']  # prompts (encoded)
        game_data['prompt'] = [dict_train_text[idx] for idx in batch_idx]       # prompts
        
        # Get the corresponding responses to the prompts
        t = time.time()
        response_tensors = []
        for i in tqdm(range(int(config['batch_size']/fbs))):
            queries = batch_dict[i*fbs:(i+1)*fbs]
            response  = respond_to_batch(gpt2_model, queries,txt_len=config['max_len'])
            response_tensors.append(response)
        
        response_tensors = torch.cat(response_tensors) # encoded responses
        game_data['response'] = [gpt2_tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]
        timing['time/get_response'] = time.time()-t

        # Empathy Scoring
        t = time.time()
        empathy_inputs, attention_masks = build_bert_batch_from_txt(game_data['response'], empathy_tokenizer, device) # tokenise inputs
        pos_logits = []
        for i in range(int(config['batch_size']/fbs)):
            res = empathy_classifier.forward(empathy_inputs[i*fbs:(i+1)*fbs],
                                        attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach() # take the logit for high empathy
            pos_logits.append(res)

        rewards = torch.cat(pos_logits)
        timing['time/get_sentiment_preds'] = time.time()-t

        # Run PPO Training 
        t = time.time()
        stats = ppo_trainer.step(batch_dict, response_tensors, rewards)
        timing['time/optimization'] = time.time()-t
        
        # Log everything
        timing['time/epoch'] = time.time()-t0
        table_rows = [list(r) for r in zip(game_data['prompt'], game_data['response'], rewards.cpu().tolist())]
        logs.update({'game_log':wandb.Table(
            columns=['prompt', 'response', 'reward'],
            rows=table_rows)})
        logs.update(timing)
        logs.update(stats)
        logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
        logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
        logs['env/reward_dist'] = rewards.cpu().numpy()
        wandb.log(logs)

# 55948: first run - warm startup (supervised)
if __name__ == "__main__":
    run_RL()