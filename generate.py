from transformers import AutoTokenizer, GPT2LMHeadModel
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

    return encoded_input 

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
encoding_list = encoded_df(df, tokenizer)

print(f'results for {PRE_TRAINED_MODEL_NAME}')
df_responses = pd.DataFrame(columns = df.columns())

for encoding in encoding_list:
    # remove [EOS] token but maintain shape
    input_ids = encoding['input_ids']
    input_ids = input_ids[0][:-1].view(1,-1) 
    output = model.generate(input_ids,
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
    
    print(output)
    decode = tokenizer.decode(output[0]) #, skip_special_tokens=True

    # break at [PAD] token
    print(decode.split('[PAD]')[0])

    # rewritings = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

    # for i, r in enumerate(rewritings):
    #    print(f"{i}: {r}")


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

