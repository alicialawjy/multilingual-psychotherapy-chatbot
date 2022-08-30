'''
Script used to obtain rewritings from the generative model.
'''

import torch
import pandas as pd
from transformers import AutoTokenizer, GPT2LMHeadModel
from torch.utils.data import TensorDataset, Dataset
from run_generation import GPT2RewritingDataset

############# Data Loader for Inference ############# 
def encoded_df_inference(df, tokenizer):
    # extract df columns
    emotion = df['emotion'].values.tolist()
    base = df['base'].values.tolist()
    
    # concatenate df columns horizontally, joining with the respective tokens
    formatted_input = []
    for (e, b) in list(zip(emotion, base)):
        input = '[HIGH]' + e + '[SEP]' + b + '[REWRITE]'
        formatted_input.append(input)

    # encode the formatted input
    encoded_input = tokenizer(formatted_input, 
                            return_tensors = 'pt',
                            padding = True,
                            truncation = True
                            )

    return emotion, base, encoded_input 

# Fix Device
GPU = True
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f"Using {device}")


PRE_TRAINED_MODEL_NAME = 'rewriting/gpt2-trl/best-model'
model = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)    

df = pd.read_csv('data/generation/inference.csv', index_col=0)                                  # DataFrame of prompts for Empathetic Rewriting 
emotion, base, encoding_list = encoded_df_inference(df, tokenizer)                              # Format and encode prompts
dataloader = GPT2RewritingDataset(tokenizer=tokenizer, encodings=encoding_list, train=False)    # Dataloader object
full_dict_list = [dataloader.__getitem__(n) for n in range(len(df))]              
full_input_ids = dataloader.collate_fn(full_dict_list)['input_ids'].to(device)                  # Collated prompts 

print(f'results for {PRE_TRAINED_MODEL_NAME}')
df_responses = pd.DataFrame(columns = ['emotion','base','rewriting'])

for (e,b,input_ids) in zip(emotion,base,full_input_ids):
    # extract only the rewritings (remove prompts)
    start_idx = len(input_ids)                              
    output = model.generate(input_ids.view(1,-1),  # must reshape
                            max_length = 100,       
                            do_sample=True, 
                            temperature=1,
                            top_k=50, 
                            top_p=0.95, 
                            num_return_sequences= 25,
                            num_beams = 10,
                            no_repeat_ngram_size = 2,
                            clean_up_tokenization_spaces = True,
                            repetition_penalty = 1.2,               # as per CTRL paper
                            early_stopping = True)
    
    rewritings = [tokenizer.decode(out[start_idx:]) for out in output]
    emo_list = [e]*len(rewritings)
    base_list = [b]*len(rewritings)
    df_base = pd.DataFrame(columns=['emotion','base','rewriting'], data=zip(emo_list,base_list,rewritings))
    # remove any duplicate sentences
    df_base = df_base.drop_duplicates(keep='first') 
    df_responses = pd.concat([df_responses, df_base],ignore_index=0)

df_responses.to_csv('data/generation/empathetic_rewritings.csv')
