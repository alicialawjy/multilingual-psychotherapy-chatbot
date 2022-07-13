from transformers import AutoTokenizer, GPT2LMHeadModel
import torch

# Fix Device
GPU = True
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f"Using {device}")

PRE_TRAINED_MODEL_NAME = 'rewriting/gpt2-supervised/25/best-model'
model = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)         

prompt = ['[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]这是由特别事件引起的吗？[REWRITE]',
        '[PROMPT]男性[SEP]18-39,悲伤[SEP]这是由最近或遥远的事件（或多个事件）引起的吗？[REWRITE]',
        '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]您最近是否尝试过协议6，并发现由于旧事件而重新点燃了无法控制的情绪？[REWRITE]',
        '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]您最近是否尝试过协议11，并发现由于旧事件而重新点燃了无法控制的情绪？[REWRITE]',
        '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]谢谢。现在我会问一些问题以了解您的情况。[REWRITE]',
        '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]您是否对某人有强烈的感受或表达以下任何情绪：[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]您认为您应该成为别人的救星吗？[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]您是否将自己视为受害者，将自己的负面情绪归咎于他人？[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]您觉得您在试图控制某人吗？[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]当出现问题时，您是否总是责怪和指责自己？[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]在之前的对话中，您是否考虑过其他提出的观点？[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]您是否正在经历个人危机（与亲人相处有困难，例如与朋友闹翻）？[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]快乐[SEP]那很好！让我推荐一个您可以尝试的协议。[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]根据您所说的，我相信您正在感受{}。这个对吗？[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]我很抱歉。请从下面的情绪中选择最能反映您感受的情绪：[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]感谢您的参与。再见[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]这是我的建议，请选择您想尝试的协议[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]请现在尝试通过此协议。完成后，按“继续”[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]采取此协议后，您感觉更好还是更糟？[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]您想尝试另一种协议吗？ （病人感觉好多了）[REWRITE]',
        '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]您想尝试另一种协议吗？ （病人感觉更糟)[REWRITE]']

for p in prompt:
    input_ids = tokenizer.encode(p, return_tensors = 'pt').to(device)
    output = model.generate(input_ids, 
                            max_length = 100, 
                            do_sample=True, 
                            temperature=1.5,
                            top_k=50, 
                            top_p=0.95, 
                            num_return_sequences= 1,
                            num_beams = 5,
                            no_repeat_ngram_size = 2,
                            clean_up_tokenization_spaces=True,
                            return_full_text=False,
                            early_stopping = True)
    print(tokenizer.decode(output, skip_special_tokens=True))
    #rewritings = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

    #for i, r in enumerate(rewritings):
    #    print(f"{i}: {r}")

# 55957
# 55959: with temp=2