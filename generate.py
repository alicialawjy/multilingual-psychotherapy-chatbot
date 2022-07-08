from transformers import AutoTokenizer, GPT2LMHeadModel
import torch

# Fix Device
GPU = True
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f"Using {device}")

PRE_TRAINED_MODEL_NAME = 'rewriting/gpt2-supervised/checkpoint-1100'
model = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)         

prompt = '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]这是由特别事件引起的吗？[REWRITE]'
input_ids = tokenizer.encode(prompt, return_tensors = 'pt').to(device)
output = model.generate(input_ids, 
                        max_length = 100, 
                        do_sample=True, 
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