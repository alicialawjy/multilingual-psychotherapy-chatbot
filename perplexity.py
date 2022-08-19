from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import numpy as np
import pandas as pd

def compute_ppl(input):
    '''
    Function used to compute fluency of a given input sentence. 
    Taken as inverse perplexity - repetition penalty
    Used during reinforcement learning.
    - encoding [list]: a list of the sentence's encoded form.
    - gpt2_eval_model [GPT2LMHeadModel object]: the gpt-2 model used to compute the fluency score.

    Returns: a fluency score [float]
    '''
    encoding = tokenizer(input, 
                        return_tensors = 'pt',
                        # padding = True,
                        # truncation = True
                        )

    with torch.no_grad():
        loss = model(input_ids=encoding.input_ids, labels=encoding.input_ids).loss

    return np.exp(loss.cpu().detach().numpy())

# Use GPU
GPU = True
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f"Using {device}")

# GPT-2 Model
PRE_TRAINED_MODEL_NAME = 'uer/gpt2-chinese-cluecorpussmall' 
model = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)    

df = pd.read_csv('data/generation/empathetic_rewritings.csv')
perplexity = [compute_ppl(x) for x in df['rewriting'].tolist()]
print('std dev = ', np.std(perplexity))
print('mean = ', np.mean(perplexity))

df['perplexity'] = perplexity
df.to_csv('rewritings_ppl.csv')