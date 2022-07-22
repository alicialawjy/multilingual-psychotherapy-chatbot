from transformers import AutoTokenizer, GPT2LMHeadModel
from torch.utils.data import TensorDataset, Dataset
import torch
import re
import pandas as pd

############# Data Loader for GPT-2 ############# 
def encoded_df(df, tokenizer):
    # extract df columns
    #gender = df['gender'].values.tolist()
    #age = df['age'].values.tolist()
    emotion = df['emotion'].values.tolist()
    base = df['base'].values.tolist()
    
    # concatenate df columns horizontally, joining with the respective tokens
    formatted_input = []
    for (e, b) in list(zip(emotion, base)):
        input = '[PROMPT]' + e + '[SEP]' + b + '[REWRITE]'
        formatted_input.append(input)

    # encode the formatted input
    encoded_input = tokenizer(formatted_input, 
                            return_tensors = 'pt',
                            padding = True,
                            truncation = True
                            )

    return emotion, base, encoded_input 

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


# Fix Device
GPU = True
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f"Using {device}")

PRE_TRAINED_MODEL_NAME = 'rewriting/gpt2-supervised-experiment0/100/best-model'
model = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)    

df = pd.read_csv('data/empathy/base_utterances/inference.csv', index_col=0)
emotion, base, encoding_list = encoded_df(df, tokenizer)
dataloader = GPT2RewritingDataset(tokenizer=tokenizer, encodings=encoding_list, train=False) # dataloader object
full_dict_list = [dataloader.__getitem__(n) for n in range(len(df))]              
full_input_ids = dataloader.collate_fn(full_dict_list)['input_ids'].to(device)   # get prompts (encoded)

print(f'results for {PRE_TRAINED_MODEL_NAME}')
df_responses = pd.DataFrame(columns = ['emotion','base','rewriting'])

for (e,b,input_ids) in zip(emotion,base,full_input_ids):
    # remove [EOS] token but maintain shape
    output = model.generate(input_ids.view(1,-1),
                            max_length = 100, 
                            do_sample=True, 
                            temperature=0.7,
                            top_k=50, 
                            top_p=0.95, 
                            num_return_sequences= 3,
                            num_beams = 5,
                            no_repeat_ngram_size = 2,
                            clean_up_tokenization_spaces=True,
                            return_full_text=False,
                            early_stopping = True)

    start_idx = torch.sum(input_ids!=0) - 1
    rewritings = [tokenizer.decode(out)[start_idx:].split('[PAD]')[0] for out in output]
    emo_list = [e]*len(rewritings)
    base_list = [b]*len(rewritings)
    df_base = pd.DataFrame(columns=['emotion','base','rewriting'], data=zip(emo_list,base_list,rewritings))
    df_responses = pd.concat([df_responses, df_base],ignore_index=0)

df_responses.to_csv('inference_results.csv')

##### experiment 0 [PROMPT] emo [SEP] base [REWRITE] #####
# 57509: best model


##### Experiment 3 [HIGH/LOW] emo [SEP] base [REWRITE]##### 
# 57043: checkpoint 34500 *best one yet
# 57165: best model (checkpoint 49500)

##### Experiment 3b emo[SEP]base[HIGH/LOW]#####
# 57166: checkpoint 30000 *not bad
# 57044: checkpoint 42000
# 57045: checkpoint 51000
# 57162: best model (checkpoint 51000)

