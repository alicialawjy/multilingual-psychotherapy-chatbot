'''
Section on Reinforcement Learning with PPO was done with reference to:
https://github.com/lvwerra/trl
'''

import os
import random
import time
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from trl.ppo import PPOTrainer
from torch.nn import functional as F
from trl.core import build_bert_batch_from_txt
from torch.utils.data import TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments, AutoModelForSequenceClassification, \
    GPT2LMHeadModel, GPT2Tokenizer


############# Data Loader for GPT-2 ############# 
def encoded_df(df, tokenizer, supervised):
    '''
    Function used to encode dataframe. 
    - df [DataFrame]: input to be encoded
    - tokenizer [AutoTokenizer object]: tokenizer used to carry out the encoding
    - supervised [bool]: 
        True if supervised learning, 
        False if reinforcement learning
    '''
    # extract df columns
    emotion = df['emotion'].values.tolist()
    base = df['base'].values.tolist()
    rewriting = df['rewriting'].values.tolist()

    if supervised:
        transformation = df['transformation'].values.tolist()

    else:
        transformation = ['']*len(base)
        semantic_label = df['semantic'].values.tolist()

    # input follows the following structure: 
    # [HIGH/LOW] emotion [SEP] base utt [REWRITE] rewriting (if supervised)
    formatted_input = []
    for (t, e, b, r) in list(zip(transformation, emotion, base, rewriting)):
        if not supervised:
            # during reinforcement learning, use only [HIGH] token
            input = '[HIGH]'

        else:
            if t == 'HIGH':
                input = '[HIGH]'
            elif t == 'LOW':
                input = '[LOW]'
            else:
                raise Exception("No transformation listed! Needed for supervised learning")
        
        input += e + '[SEP]' + b + '[REWRITE]'

        # if training for supervised learning (i.e. not inference/ RL), append the rewritings as well
        if supervised:
            input += r
        
        formatted_input.append(input)

    # encode the formatted input
    encoded_input = tokenizer(formatted_input, 
                            return_tensors = 'pt',
                            padding = True,
                            truncation = True
                            )

    if supervised:
        return encoded_input 
    else:
        return formatted_input, semantic_label, encoded_input

class GPT2RewritingDataset(Dataset):
    ''' 
    DataLoader for GPT-2 Rewriting Task.

    '''
    def __init__(self, tokenizer, encodings, supervised): 
        self.encodings = encodings
        self.tokenizer = tokenizer
        self.supervised = supervised

    def __len__(self): 
        '''
        Obtains length of the input.
        '''
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx): 
        '''
        Obtains the labels, input_ids and attention_mask corresponding to a particular index (idx).
        '''
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]

        if not self.supervised:
            # if for reinforcement learning, remove the EOS token that automatically gets added by the encoder
            last_idx = torch.sum(attention_mask) - 1
            input_ids[last_idx] = 0 
            attention_mask[last_idx] = 0

        return {'labels': input_ids, 'input_ids': input_ids, 'attention_mask': attention_mask}
        
    # Takes batches of the input data
    def collate_fn(self, batch):
        '''
        Collates batches of the input data. Items in batch are obtained via __getitem__.
        '''
        # format into tensors
        input_ids = []
        attention_mask = []
        token_type_ids = []
        labels = []
        for b in batch:
            input_ids.append(list(b['input_ids']))
            attention_mask.append(list(b['attention_mask']))
            labels.append(list(b['labels']))
        
        return {'labels': torch.tensor(labels), 'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}


############# Fluency computation ############# 
def compute_fluency(encoding, gpt2_eval_model):
    '''
    Function used to compute fluency of a given input sentence. 
    Taken as inverse perplexity - repetition penalty
    Used during reinforcement learning.
    - encoding [list]: a list of the sentence's encoded form.
    - gpt2_eval_model [GPT2LMHeadModel object]: the gpt-2 model used to compute the fluency score.

    Returns: a fluency score [float]
    '''
    with torch.no_grad():
        loss = gpt2_eval_model(input_ids=encoding, labels=encoding).loss

    perplexity = np.exp(loss.cpu().detach().numpy())

    token_seen = []
    repetition_penalty = 0 # RP
    last_seen = 0
    for token in encoding:
        # if token not seen before, no RP
        if token not in token_seen:
            last_seen = 1
            token_seen.append(token)

        # if seen and was the same as the last_seen token (i.e. repeated the same token consecutively)
        elif token in token_seen[-1]:
            last_seen +=1
            repetition_penalty += 0.01*last_seen # compound the penalty
        
        # seen but not repeated
        else:
            # if not repeated but previously seen
            last_seen = 1
            repetition_penalty += 0.01

    return 1/perplexity - repetition_penalty


############# Supervised Learning (Warmup) Main Code ############# 
def run_supervised():
    # Directory in which to save the model checkpoints
    main_dir = 'rewriting/gpt2-supervised'   
    os.environ["WANDB_DISABLED"] = "true"

    # Fix Device
    GPU = True
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using {device}")

    ##### G P T - 2 #####
    # GPT-2 Model
    PRE_TRAINED_MODEL_NAME = 'uer/gpt2-chinese-cluecorpussmall' 
    model = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)

    # GPT-2 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)               
    # additional tokens for conditional generation  
    additional_tokens = {'high_token':'[HIGH]', 'low_token':'[LOW]', 'rewrite_token':'[REWRITE]'}
    tokenizer.add_tokens(list(additional_tokens.values()), special_tokens=True)     # add into tokenizer vocabulary (found in added_tokens.json)
    for token_name, token in additional_tokens.items():
        setattr(tokenizer, token_name, token)                                       # assign corr. names (used in dataloader)

    model.resize_token_embeddings(len(tokenizer))                                   # resize the model token embedding space
    tokenizer.save_pretrained(main_dir)                                        # save the new tokenizer in the model directory

    ##### D A T A S E T S #####
    # DataFrames
    df_supervised = pd.read_csv('data/generation/generation.csv',index_col=0).sample(frac=1)
    df_train, df_val = train_test_split(df_supervised, test_size=0.2, shuffle=True, random_state=0)

    # Format and encode df with encoded_df()
    dict_train = encoded_df(df=df_train, supervised=True, tokenizer=tokenizer)
    dict_val = encoded_df(df=df_val, supervised=True, tokenizer=tokenizer) 

    # Get DataLoader object, used by Trainer
    dataloader_train = GPT2RewritingDataset(tokenizer=tokenizer, encodings=dict_train, supervised=True) 
    dataloader_val = GPT2RewritingDataset(tokenizer=tokenizer, encodings=dict_val, supervised=True) 

    ##### T R A I N I N G #####
    # Early Stopping Module
    trainer_callback = EarlyStoppingCallback(early_stopping_patience = 20,
                                            early_stopping_threshold = 0.001)

    # Training Arguments
    training_args = TrainingArguments(output_dir = main_dir,                # Output directory where checkpoints + models are saved
                                    overwrite_output_dir = True,            # Overwrite the output directory if populated
                                    learning_rate = 1e-5,                   # Learning rate
                                    num_train_epochs = 1,                  # Number of training epochs
                                    warmup_steps = 100,                     # Number of warmup steps
                                    per_device_train_batch_size = 4,        # Batch size for training
                                    # Early Stopping Arguments
                                    per_device_eval_batch_size = 4,         # Batch size for evaluation
                                    evaluation_strategy = 'steps',          # Number of update steps between two evaluations
                                    eval_steps = 500,                       # Evaluate every 500 steps
                                    save_strategy = 'steps',                # Save strategy
                                    save_steps = 1500,                      # Save every 1500 steps
                                    save_total_limit = 5,                   # Save only the 5 latest models. Deletes older models.
                                    logging_strategy = 'steps',             # Logging strategy
                                    logging_dir = f'{main_dir}/logs',       # Directory where logs are saved
                                    logging_steps = 500,                    # Log every 500 steps
                                    include_inputs_for_metrics = True,      # Include inputs for metrics
                                    metric_for_best_model = 'eval_loss',    # Decide best model based on the on with the lowest eval_loss
                                    greater_is_better = False,              # Lower eval_loss is better
                                    load_best_model_at_end = True           # Required by EarlyStoppingCallback
                                    )

    # Trainer
    trainer = Trainer(args=training_args,                                   # Training Arguments
                    model = model,                                          # Model
                    tokenizer = tokenizer,                                  # Tokenizer
                    train_dataset = dataloader_train,                       # Training dataset
                    eval_dataset = dataloader_val,                          # Validation dataset
                    data_collator = dataloader_train.collate_fn,            # DataCollator
                    callbacks = [trainer_callback]                          # EarlyStoppingCallback module
                    )

    trainer.train()

    # As we enabled load_best_model_at_end, 
    # this will save the best model and its tokenizer
    trainer.save_model(f'{main_dir}/best-model')
    tokenizer.save_pretrained(f'{main_dir}/best-model')

    print(f'Training Completed! See {main_dir}/best-model for the best-model with lowest eval_loss')


############# Reinforcement Learning Main Code ############# 
def run_RL():
    ##### P A R A M E T E R S ######
    config = {
        "lm_name": 'rewriting/gpt2-supervised/best-model',                      # gpt-2 supervised-warmup model used for generation
        "lm_eval_name": 'uer/gpt2-chinese-cluecorpussmall',                     # gpt-2 model used to compute perplexity
        "empathy_classifier_name": "empathy_classifier/binary-empathy",         # empathy classifier (xlm-r)
        "semantic_classifier_name": "semantic_classifier/4e05/best-model",      # semantic classifier (xlm-r) 
        "steps": 10000,                                                                      
        "batch_size": 16, # 2
        "forward_batch_size": 4, # 2
        "ppo_epochs": 4,
        "max_len": 50,
        "lr": 5e-6,
        "init_kl_coef":0.2,
        "seed": 1,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1, 
        "empathy_weight": 4,        
        "semantic_weight": 0.25,
        "fluency_weight": 1
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

    # Semantic Classifier
    SEMANTIC_CLASSIFIER_NAME = config['semantic_classifier_name']
    semantic_classifier = AutoModelForSequenceClassification.from_pretrained(SEMANTIC_CLASSIFIER_NAME).to(device)
    semantic_tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_CLASSIFIER_NAME)                    # shared with empathy classifier as well

    # GPT2 Language Models
    GPT2_PRETRAINED_NAME = config['lm_name']
    gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_PRETRAINED_NAME)                            # gpt2 tokenizer - shared amongst all models
    gpt2_model = GPT2HeadWithValueModel.from_pretrained(GPT2_PRETRAINED_NAME).to(device)            # model to be finetuned
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(GPT2_PRETRAINED_NAME).to(device)        # reference model for PPO

    GPT2_EVAL_PRETRAINED_NAME = config['lm_eval_name']
    gpt2_eval_model = GPT2LMHeadModel.from_pretrained(GPT2_EVAL_PRETRAINED_NAME).to(device)         # model for fluency evaluation
    gpt2_eval_model.resize_token_embeddings(len(gpt2_tokenizer))                                    # resize the eval model's embedding space to the new tokenizer (has additional tokens)
    
    wandb.watch(gpt2_model, log='all')

    ##### L O A D  D A T A S E T S #####
    df = pd.read_csv('data/generation/generation.csv', index_col=0).sample(frac=1)                                      # DataFrame
    train_text, semantic_label, dict_train_encoded = encoded_df(df=df, supervised=False, tokenizer=gpt2_tokenizer)      # Format and encode
    train_dataloader = GPT2RewritingDataset(tokenizer=gpt2_tokenizer, encodings=dict_train_encoded, supervised=False)   # Dataloader object
    
    ##### P P O  R L  T R A I N I N G  L O O P #####
    ppo_trainer = PPOTrainer(model=gpt2_model, ref_model=gpt2_model_ref, tokenizer=gpt2_tokenizer, **config)
    fbs = config['forward_batch_size']  
    mean_max = 0
    stdev_min = 100 # set a large number
    for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
        torch.cuda.empty_cache()
        logs = dict()
        game_data = dict()
        timing = dict()
        t0 = time.time()
        
        # Batch prompts
        batch_idx = random.sample(range(train_dataloader.__len__()),k=config['batch_size'])
        batch_dict_list = [train_dataloader.__getitem__(n) for n in batch_idx]              
        batch_dict = train_dataloader.collate_fn(batch_dict_list)['input_ids'].to(device)   # prompts (encoded)
        game_data['prompt'] = [train_text[idx] for idx in batch_idx]                        # prompts
        batch_semantic_label = [semantic_label[idx] for idx in batch_idx]                   # semantic label corr to the prompt
        
        # Get the corresponding responses to the prompts
        t = time.time()
        response_tensors = []
        for i in range(int(config['batch_size']/fbs)):
            queries = batch_dict[i*fbs:(i+1)*fbs].to(device)
            response = respond_to_batch(gpt2_model, queries, txt_len=config['max_len'])
            response_tensors.append(response)
        
        # Encoded responses
        response_tensors = torch.cat(response_tensors).to(device) 
        game_data['response'] = [gpt2_tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]
        timing['time/get_response'] = time.time()-t

        # REWARD SCORING
        t = time.time()
        classifier_inputs, attention_masks = build_bert_batch_from_txt(game_data['response'], semantic_tokenizer, device) # tokenise inputs for classifiers
        rewards = []
        empathy = []
        semantic = []
        fluency = []
        for i in range(int(config['batch_size']/fbs)):
            # empathy score - take the logit corr to high empathy ([:,1])
            empathy_score = empathy_classifier.forward(classifier_inputs[i*fbs:(i+1)*fbs],
                                                        attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()   
            # semantic score - take the logit for the corr semantic
            semantic_score_all = semantic_classifier.forward(classifier_inputs[i*fbs:(i+1)*fbs],
                                                        attention_masks[i*fbs:(i+1)*fbs])[0].detach()   # this is shape (batch_size x num_of_semantic_labels=20)
            semantic_score = [logits[idx] for (logits, idx) in zip(semantic_score_all, batch_semantic_label[i*fbs:(i+1)*fbs])]
            # fluency score 
            with torch.no_grad():
                fluency_score = [compute_fluency(encoding, gpt2_eval_model) for encoding in response_tensors[i*fbs:(i+1)*fbs]]

            # total score
            w_e = config['empathy_weight']
            w_s = config['semantic_weight']
            w_f = config['fluency_weight']
            total_score = [w_e * e + w_s * s + w_f * f for (e,s,f) in zip(empathy_score, semantic_score, fluency_score)] 

            # convert list of tensors into a single tensor and append
            empathy.append(empathy_score)
            semantic.append(torch.stack(semantic_score))
            fluency.append(torch.tensor(fluency_score))
            rewards.append(torch.stack(total_score))
        
        empathy = torch.cat(empathy)
        semantic = torch.cat(semantic)
        fluency = torch.cat(fluency)
        rewards = torch.cat(rewards).to(device)
        timing['time/get_sentiment_preds'] = time.time()-t

        # Run PPO Training 
        t = time.time()
        stats = ppo_trainer.step(batch_dict, response_tensors, rewards)
        timing['time/optimization'] = time.time()-t
        
        # Log everything
        timing['time/epoch'] = time.time()-t0
        table_rows = [list(r) for r in zip(game_data['prompt'], game_data['response'], empathy, semantic, fluency, rewards.cpu().tolist())] # removed fluency
        logs.update({'game_log':wandb.Table(
            columns=['prompt', 'response', 'empathy_logit','sematic_logit','fluency','reward'], # removed inv_ppl
            rows=table_rows)})
        logs.update(timing)
        logs.update(stats)
        reward_mean = torch.mean(rewards).cpu().numpy()
        reward_std = torch.std(rewards).cpu().numpy()
        logs['env/reward_mean'] = reward_mean
        logs['env/reward_std'] = reward_std
        logs['env/reward_dist'] = rewards.cpu().numpy()

        wandb.log(logs)

        # save if a better checkpoint observed
        if reward_mean > mean_max or reward_std < stdev_min: 
            # if only one of the metrics are better, save for consideration
            output_dir = f"rewriting/gpt2-trl/{epoch}"
            gpt2_model.save_pretrained(output_dir)
            gpt2_tokenizer.save_pretrained(output_dir)
            
            if reward_mean >= mean_max and reward_std <= stdev_min: 
                # only replace the mean and std dev if both beaten 
                mean_max = reward_mean 
                stdev_min = reward_std


#### Code to execute #####
if __name__ == "__main__":
    run_supervised()
    run_RL()

