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
from torch.nn import functional as F

############# Data Loader for GPT-2 ############# 
def encoded_df(df, tokenizer, supervised, train=False):
    '''
    supervised (bool): true if supervised learning, false if reinforcement learning
    train (bool): true if training dataset, false is test/validation. only relevant for supervised learning.
    '''
    # extract df columns
    #gender = df['gender'].values.tolist()
    #age = df['age'].values.tolist()
    emotion = df['emotion'].values.tolist()
    base = df['base'].values.tolist()
    rewriting = df['rewriting'].values.tolist()
    transformation = df['transformation'].values.tolist()

    if not supervised:
        semantic_label = df['semantic'].values.tolist()
        transformation_label = df['transformation_label'].values.tolist()
    
    # # concatenate df columns horizontally, joining with the respective tokens
    # formatted_input = []
    # for (g, a, e, b, r) in list(zip(gender, age, emotion, base, rewriting)):
    #     input = '[PROMPT]' + g + '[SEP]' + a + '[SEP]' + e + '[SEP]' + b + '[REWRITE]'
    #     # if supervised, append the rewritings as well
    #     if supervised:
    #         input += r
        
    #     formatted_input.append(input)

    # EXPERIMENT 3
    formatted_input = []
    for (t, e, b, r) in list(zip(transformation, emotion, base, rewriting)):

        input = ''

        # instead of [REWRITE], we use [HIGH] or [LOW]
        if t == 'HIGH':
            input += '[HIGH]'
        elif t == 'LOW':
            input += '[LOW]'
        else:
            raise Exception("No transformation listed!")
        
        input += e + '[SEP]' + b + '[REWRITE]'

        # if training for supervised learning (i.e. not inference/ RL), append the rewritings as well
        if supervised and train:
            input += r
        
        formatted_input.append(input)
    
    # # EXPERIMENT 3B
    # formatted_input = []
    # for (t, e, b, r) in list(zip(transformation, emotion, base, rewriting)):

    #     input = e + '[SEP]' + b 
    #     # instead of [REWRITE], we use [HIGH] or [LOW]
    #     if t == 'HIGH':
    #         input += '[HIGH]'
    #     elif t == 'LOW':
    #         input += '[LOW]'
    #     else:
    #         raise Exception("No transformation listed!")

    #     # if training dataset for supervised learning, append the rewritings as well
    #     if supervised and train:
    #         input += r
        
    #     formatted_input.append(input)

    # encode the formatted input
    encoded_input = tokenizer(formatted_input, 
                            return_tensors = 'pt',
                            padding = True,
                            truncation = True
                            )

    if supervised:
        return encoded_input 
    else:
        return formatted_input, semantic_label, transformation_label, encoded_input

class GPT2RewritingDataset(Dataset):
    ''' 
    DataLoader for GPT-2 Rewriting Task 

    '''
    def __init__(self, tokenizer, encodings, train=True): # ok
        self.encodings = encodings
        self.tokenizer = tokenizer
        self.train = train

    def __len__(self): # ok
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx): 
        # # need to pass items to collate_fn with the following acceptable keys:
        # # ['input_ids', 'past_key_values', 'attention_mask', 'token_type_ids', 'position_ids', 'head_mask', 
        # # 'inputs_embeds','encoder_hidden_states', 'encoder_attention_mask', 'labels', 'use_cache', 
        # # 'output_attentions', 'output_hidden_states','return_dict', 'labels', 'label', 'label_ids']
        input_ids = self.encodings['input_ids'][idx]
        attention_mask = self.encodings['attention_mask'][idx]

        if not self.train:
            # if not for training, remove the EOS token that automatically gets added by the encoder
            last_idx = torch.sum(attention_mask) - 1
            input_ids[last_idx] = 0 
            attention_mask[last_idx] = 0

        return {'labels': input_ids, 'input_ids': input_ids, 'attention_mask': attention_mask}
        
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
            labels.append(list(b['labels']))
        
        return {'labels': torch.tensor(labels), 'input_ids': torch.tensor(input_ids), 'attention_mask': torch.tensor(attention_mask)}

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

# def compute_metrics(eval_predictions):
#     # Eval_predictions gives the logits and loss of the predicted words
#     # Convert predictions into their textual representations
#     response = eval_predictions.predictions
#     PRE_TRAINED_MODEL_NAME = 'uer/gpt2-chinese-cluecorpussmall' # 'rewriting/gpt2-supervised-transformation-v2/100'
#     gpt2tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#     all_input_ids = []
#     for r in response:
#         input_ids = []
#         for word in r:
#             # Get the logits
#             logits = torch.unsqueeze(torch.tensor(word),dim=0)
#             # Greedy search: take most likely
#             next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
#             # Append to input_ids
#             input_ids.append(next_token)

#         input_ids = torch.squeeze(torch.stack(input_ids))
#         all_input_ids.append(input_ids)

#     all_input_ids=torch.stack(all_input_ids)

#     # Get the scores per textual output
#     CLASSIFIER_MODEL_NAME = 'saved_models/Knowledge Distillation/2nd-tune-lr1e05-temp5'
#     classifiertokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
#     empathy_classifier =  AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL_NAME).to(device)

#     score = 0
#     for text in all_input_ids:
#         gpt_decoded = gpt2tokenizer.decode(text[0])
#         classifier_encoded =  classifiertokenizer(gpt_decoded, padding=True, truncation=True, return_tensors='pt')
#         empathy_logits = 

#     print('decoded pred:', )

############# Fluency computation ############# 
def compute_fluency(encoding, gpt2_eval_model):
    with torch.no_grad():
        loss = gpt2_eval_model(input_ids=encoding, labels=encoding).loss

    perplexity = np.exp(loss.detach().numpy())

    token_seen = []
    repetition_penalty = 0
    for token in encoding:
        if token in token_seen:
            repetition_penalty += 0.01
        else:
            token_seen.append(token)
    
    fluency = 1/perplexity - repetition_penalty

    return fluency

############# Supervised Learning Main Code ############# 
def run_supervised():
    main_dir = 'rewriting/gpt2-supervised-experiment3b/50'
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
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)               # model tokenizer 
    # additional tokens for conditional generation
    # additional_tokens = {'rewrite_token':'[REWRITE]', 'prompt_token':'[PROMPT]'}  
    # additional_tokens = {'high_token':'[HIGH]', 'low_token':'[LOW]', 'rewrite_token':'[REWRITE]'}
    additional_tokens = {'high_token':'[HIGH]', 'low_token':'[LOW]'}
    tokenizer.add_tokens(list(additional_tokens.values()), special_tokens=True)     # add into tokenizer vocabulary (found in added_tokens.json)
    for token_name, token in additional_tokens.items():
        setattr(tokenizer, token_name, token)                                       # assign corr. names (used in dataloader)

    model.resize_token_embeddings(len(tokenizer))                                   # resize the model token embedding space
    tokenizer.save_pretrained(f'{main_dir}')                                        # save the new tokenizer in the model directory

    ##### D A T A S E T S #####
    # DataFrames
    df_supervised = pd.read_csv('data/empathy/ex3-low_high_noextra/experiment3_train.csv',index_col=0).sample(frac=1)
    df_train, df_val = train_test_split(df_supervised, test_size=0.2, shuffle=True, random_state=0)

    # Format and encode df with encoded_df()
    dict_train = encoded_df(df=df_train, supervised=True, train=True, tokenizer=tokenizer)
    dict_val = encoded_df(df=df_val, supervised=True, train=True, tokenizer=tokenizer) 

    # Get DataLoader object, used by Trainer
    dataloader_train = GPT2RewritingDataset(tokenizer=tokenizer, encodings=dict_train, train=True) 
    dataloader_val = GPT2RewritingDataset(tokenizer=tokenizer, encodings=dict_val, train=True) 

    ##### T R A I N I N G #####
    # Early Stopping Module
    # trainer_callback = EarlyStoppingCallback(early_stopping_patience = 20,
    #                                         early_stopping_threshold = 0.001)

    # Training Arguments
    training_args = TrainingArguments(output_dir = main_dir,                # Output directory where checkpoints + models are saved
                                    overwrite_output_dir = True,            # Overwrite the output directory if populated
                                    learning_rate = 1e-5,                   # Learning rate
                                    num_train_epochs = 50,                  # Number of training epochs
                                    warmup_steps = 100,
                                    per_device_train_batch_size = 4,        # Batch size for training
                                    # Early Stopping Arguments
                                    per_device_eval_batch_size = 4,         # Batch size for evaluation
                                    evaluation_strategy = 'steps',          # Number of update steps between two evaluations
                                    eval_steps = 500,                       # Evaluate every 50 steps
                                    save_strategy = 'steps',                # Save strategy
                                    save_steps = 1500,                      # Save every 500 steps
                                    # save_total_limit = 50,                  # Save only the 5 latest models. Deletes older models
                                    logging_strategy = 'steps',             # Logging strategy
                                    logging_dir = f'{main_dir}/logs',
                                    logging_steps = 500,                    # Log every 100 steps
                                    include_inputs_for_metrics = True,
                                    metric_for_best_model = 'eval_loss',    # Decide based on eval_loss
                                    greater_is_better = False,              # Lower eval_loss is better
                                    load_best_model_at_end = True           # Required by EarlyStoppingCallback
                                    )

    # Trainer
    trainer = Trainer(args=training_args,
                    model = model,
                    tokenizer = tokenizer,
                    train_dataset = dataloader_train,
                    eval_dataset = dataloader_val,
                    data_collator = dataloader_train.collate_fn,
                    #compute_metrics = compute_metrics,                     # needed by Trainer.evaluate
                    #callbacks = [trainer_callback]                         # EarlyStoppingCallback module
                    )

    trainer.train()

    # Save Model and tokeniser
    trainer.save_model(f'{main_dir}/best-model')
    tokenizer.save_pretrained(f'{main_dir}/best-model')                      # save the new tokenizer in the model directory

    print(f'Training Completed! See {main_dir}/best-model for the best-model with lowest eval_loss')

############# Reinforcement Learning Main Code ############# 
def run_RL():
    ##### P A R A M E T E R S ######
    config = {
        "lm_name": 'rewriting/gpt2-supervised-experiment3/50/checkpoint-34500',         # generative model (gpt2) 
        "lm_eval_name": 'uer/gpt2-chinese-cluecorpussmall',                             # gpt to compute perplexity
        "empathy_classifier_name": "empathy_classifier/binary-empathy",                 # empathy classifier (xlm-r)
        "semantic_classifier_name": "semantic_classifier/4e05/best-model",              # semantic classifier (xlm-r) "saved_models/Emotion Classifier/2-tuned", 
        "steps": 10000,                                                                      
        "batch_size": 32, # 2
        "forward_batch_size": 8, # 2
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
        "empathy_weight": 2,        # logits range from 0 - 0.9
        "semantic_weight": 0.25,    # logits range from 0 - 20
        "fluency_weight": 2         
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

    # Semantic Classifier
    SEMANTIC_CLASSIFIER_NAME = config['semantic_classifier_name']
    semantic_classifier = AutoModelForSequenceClassification.from_pretrained(SEMANTIC_CLASSIFIER_NAME).to(device)
    semantic_tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_CLASSIFIER_NAME)

    # # GPT2 Language Models
    GPT2_PRETRAINED_NAME = config['lm_name']
    gpt2_model = GPT2HeadWithValueModel.from_pretrained(GPT2_PRETRAINED_NAME).to(device)            # model to be finetuned
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(GPT2_PRETRAINED_NAME).to(device)        # reference model
    gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_PRETRAINED_NAME)                            # gpt2 tokenizer

    GPT2_EVAL_PRETRAINED_NAME = config['lm_eval_name']
    gpt2_eval_model = GPT2LMHeadModel.from_pretrained(GPT2_EVAL_PRETRAINED_NAME).to(device)  # model for fluency evaluation
    gpt2_eval_tokenizer = AutoTokenizer.from_pretrained(GPT2_EVAL_PRETRAINED_NAME)                  # gpt2 tokenizer
    gpt2_eval_model.resize_token_embeddings(len(gpt2_eval_tokenizer))                               # resize the model token embedding space
    
    wandb.watch(gpt2_model, log='all')

    ##### L O A D  D A T A S E T S #####
    df = pd.read_csv('data/empathy/ex3-low_high_noextra/experiment3_train.csv', index_col=0).sample(frac=1) # DataFrame
    train_text, semantic_label, transformation_label, dict_train_encoded = encoded_df(df=df, supervised=False, train=False, tokenizer=gpt2_tokenizer) # format and encode
    train_dataloader = GPT2RewritingDataset(tokenizer=gpt2_tokenizer, encodings=dict_train_encoded, train=False) # dataloader object
    
    ##### P P O  R L  T R A I N I N G  L O O P #####
    ppo_trainer = PPOTrainer(model=gpt2_model, ref_model=gpt2_model_ref, tokenizer=gpt2_tokenizer, **config)
    fbs = config['forward_batch_size']  # forward batch size
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
        batch_transformation_label = [transformation_label[idx] for idx in batch_idx]    
        
        # Get the corresponding responses to the prompts
        t = time.time()
        response_tensors = []
        for i in range(int(config['batch_size']/fbs)):
            queries = batch_dict[i*fbs:(i+1)*fbs].to(device)
            response = respond_to_batch(gpt2_model, queries, txt_len=config['max_len'])
            response_tensors.append(response)
        
        response_tensors = torch.cat(response_tensors).to(device) # encoded responses
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
            # empathy score - take the logit for the corr transformation 
            empathy_score_all = empathy_classifier.forward(classifier_inputs[i*fbs:(i+1)*fbs],
                                                        attention_masks[i*fbs:(i+1)*fbs])[0].detach()   # this is shape (batch_size x num_of_empathy_labels=2)
            empathy_score = [logits[idx] for (logits, idx) in zip(empathy_score_all, batch_transformation_label[i*fbs:(i+1)*fbs])]
            # semantic score - take the logit for the corr semantic
            semantic_score_all = semantic_classifier.forward(classifier_inputs[i*fbs:(i+1)*fbs],
                                                        attention_masks[i*fbs:(i+1)*fbs])[0].detach()   # this is shape (batch_size x num_of_semantic_labels=20)
            semantic_score = [logits[idx] for (logits, idx) in zip(semantic_score_all, batch_semantic_label[i*fbs:(i+1)*fbs])]
            # fluency score = inverse perplexity - repetition penalty
            with torch.no_grad():
                fluency_score = [compute_fluency(encoding, gpt2_eval_model) for encoding in response_tensors[i*fbs:(i+1)*fbs]]
            print(fluency_score)

            # total score - multiply both logits by w_e, w_s = 2 (hyperparam w_e*e + w_s*s)
            w_e = config['empathy_weight']
            w_s = config['semantic_weight']
            w_f = config['fluency_weight']
            total_score = [e * w_e + s * w_s + f * w_f for (e,s,f) in zip(empathy_score, semantic_score, fluency_score)] # removed fluency_score

            # convert list of tensors into a single tensor and append
            empathy.append(torch.stack(empathy_score))
            semantic.append(torch.stack(semantic_score))
            fluency.append(torch.stack(fluency_score))
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
        trans = {'HIGH':0, 'LOW':1}
        for trans,trans_label in trans.items():
            key = 'env/reward_' + trans
            logs[key] = np.mean([r for r, t in zip(logs['env/reward_dist'], batch_transformation_label) if t==trans_label])
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
    run_RL()


##### LOGS #####
# data/empathy/low-high-empathy-7253-v2.csv leave for 50 epochs with 20 patience
# 56749: epoch = 50 >> trained for 13.44 epochs
# 56759: continue to train for 20 epochs lr = 5e-05
# 56770: 56759 but with smaller lr 3.5e-05
# 567774: full 50 epochs
# 56775: 100 epochs 
# 56980: experiment 3 50 epochs
# 57002: experiment 3b: no rewrite and prompt tokens, only [HIGH] and [LOW] @ 50 EPOCHS

##### REINFORCEMENT LEARNING RUNS #####
# 56175: first run with rewards * 1
#   https://wandb.ai/alicialawjy/satbot/runs/goxl4q7m?workspace=user-alicialawjy
# 56181: use rewards *2
#   https://wandb.ai/alicialawjy/satbot/runs/31ar6kcy
# 56200: use semantic classifier * 2 + empathy * 2 for rewards
#   https://wandb.ai/alicialawjy/satbot/runs/23gngqt6
# 56252: semantic + empathy + fluency
#   https://wandb.ai/alicialawjy/satbot/runs/3tfhoa2w?workspace=user-alicialawjy
# 56283: include 2144 empathetic datasets
#   https://wandb.ai/alicialawjy/satbot/runs/1ldhy878
# 56286: feed only base utterances
#   https://wandb.ai/alicialawjy/satbot/runs/242vjtvj?workspace=user-alicialawjy
# 56302: we = 2, ws = 0.1, no fluency, target KL = 3 (half initial) w/ base utt only
#   https://wandb.ai/alicialawjy/satbot/runs/2a9cy3wf
# 56325: we = 4, ws = 0.25, no fluency, target KL = 3 w/ base utt only
#   https://wandb.ai/alicialawjy/satbot/runs/1gpd4d4a
# 56589: we = 4, ws = 0.4, no fluency, target KL = 3 w/ summarised base utt (model is exploiting empathy and not generating sensible sentences)
#   https://wandb.ai/alicialawjy/satbot/runs/2f2nwgdj
# 56618: lower empathy weightage (2)
#   https://wandb.ai/alicialawjy/satbot/runs/1va386cz
# 56863: with the 45000checkpoint model
#   https://wandb.ai/alicialawjy/satbot/runs/4yy2ql23
# 56901: with the 4500 checkpoint model
#   https://wandb.ai/alicialawjy/satbot/runs/2lhyoc29
# 57407: experiment 3 w/ fluency score
#   