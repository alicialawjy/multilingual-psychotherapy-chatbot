from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import re

# Fix Device
GPU = True
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f"Using {device}")

PRE_TRAINED_MODEL_NAME = 'rewriting/gpt2-supervised-experiment3b/50/checkpoint-42000'
model = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)    

prompt = ['悲伤[SEP]这是由特别事件引起的吗？[HIGH]',
        '悲伤[SEP]非常感谢你让我知道。你会说是最近或遥远的事件（或事件）引起了这种情绪吗？[LOW]',
        '悲伤[SEP]您最近是否尝试过协议6，并发现由于旧事件而重新点燃了无法控制的情绪？[HIGH]',
        '悲伤[SEP]谢谢你，也许协议 11 可以帮助你处理这个事件，但是在我推荐它之前，我想确保你过去没有遇到过糟糕的经历。你是否已经尝试过协议 11，如果是，它是否激起了困难或创伤性的情绪？[LOW]',
        '悲伤[SEP]谢谢。现在我会问一些问题以了解您的情况。[HIGH]',
        '悲伤[SEP]我知道这可能很困难，但如果您能与我分享这一点，我会非常感激。请问您是否对某人表达或感受到了以下强烈的情绪：[LOW]',
        '悲伤[SEP]您认为您应该成为别人的救星吗？[HIGH]',
        '悲伤[SEP]我想知道你对你在这个角色中的角色有什么看法。在目前的情况下，您是否觉得自己是一个受害者，想要将自己的负面情绪转移到其他人身上？[LOW]',
        '悲伤[SEP]您觉得您在试图控制某人吗？[HIGH]',
        '悲伤[SEP],嗯，我明白了，谢谢你花时间回答所有这些问题，我仍在努力找出最好的帮助方式。你会说当出现问题时，你有把矛头指向自己的倾向吗？[LOW]',
        '悲伤[SEP]在之前的对话中，您是否考虑过其他提出的观点？[HIGH]',
        '悲伤[SEP]很抱歉让您感到难过，为了帮助您，我想深入挖掘一下。如果你不介意我问一下，你正在经历一场艰难的个人危机吗？[LOW]',
        '快乐[SEP]那很好！让我推荐一个您可以尝试的协议。[HIGH]',
        '所有情绪[SEP]感谢您通过分享您的感受来信任我。从你告诉我的情况来看，我相信你有 {} 的感觉？[LOW]',
        '所有情绪[SEP]我很抱歉。请从下面的情绪中选择最能反映您感受的情绪：[HIGH]',
        '所有情绪[SEP]再见,我要感谢您今天与我分享您的想法和感受。我真的很想很快再见到你，所以请你有空的时候过来看看。[LOW]',
        '所有情绪[SEP]这是我的建议，请选择您想尝试的协议[HIGH]',
        '所有情绪[SEP]太好了，我很高兴您愿意尝试一下，希望您会发现它很有帮助。请在准备好后仔细阅读协议，如果完成后可以按“继续”，那就太好了。[LOW]',
        '所有情绪[SEP]采取此协议后，您感觉更好还是更糟？[HIGH]',
        '所有情绪[SEP]非常好。如您所知，列表中还有更多协议，您是否认为您可能想尝试其他协议，看看情况如何？[LOW]',
        '所有情绪[SEP]您想尝试另一种协议吗？ （病人感觉更糟)[HIGH]']

print(f'results for {PRE_TRAINED_MODEL_NAME}')
for p in prompt:
    input_ids = tokenizer.encode(p, return_tensors = 'pt').to(device)
    input_ids = input_ids[0][:-1].view(1,-1) # remove [EOS] token but maintain shape

    output = model.generate(input_ids,
                            max_length = 100, 
                            do_sample=True, 
                            temperature=0.7,
                            top_k=50, 
                            top_p=0.95, 
                            num_return_sequences= 1,
                            num_beams = 5,
                            no_repeat_ngram_size = 2,
                            clean_up_tokenization_spaces=True,
                            return_full_text=False,
                            early_stopping = True)
    
    decode = tokenizer.decode(output[0]) #, skip_special_tokens=True

    # break at [PAD] token
    print(decode.split('[PAD]')[0])

    # rewritings = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

    # for i, r in enumerate(rewritings):
    #    print(f"{i}: {r}")

def hide():
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

    prompt = ['[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]特别事[REWRITE]',
            '[PROMPT]男性[SEP]18-39,悲伤[SEP]最近或遥远[REWRITE]',
            '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]协议x[REWRITE]',
            '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]问一些问题[REWRITE]',
            '[PROMPT]男性[SEP]18-39[SEP]悲伤[SEP]以下情绪[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]救星[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]受害者[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]控制某人[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]责怪自己[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]考虑其他[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]悲伤[SEP]个人危机[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]快乐[SEP]推荐协议[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]感受{}[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]选择情绪[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]感谢再见[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]选择协议[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]尝试协议按“继续”[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]更好还是更糟[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]尝试另一协议（好）[REWRITE]',
            '[PROMPT]女性[SEP]18-39[SEP]所有情绪[SEP]尝试另一协议（糟）[REWRITE]']

    prompt = ['[HIGH]悲伤[SEP]这是由特别事件引起的吗？[REWRITE]',
        '[LOW]悲伤[SEP]非常感谢你让我知道。你会说是最近或遥远的事件（或事件）引起了这种情绪吗？[REWRITE]',
        '[HIGH]悲伤[SEP]您最近是否尝试过协议6，并发现由于旧事件而重新点燃了无法控制的情绪？[REWRITE]',
        '[LOW]悲伤[SEP]谢谢你，也许协议 11 可以帮助你处理这个事件，但是在我推荐它之前，我想确保你过去没有遇到过糟糕的经历。你是否已经尝试过协议 11，如果是，它是否激起了困难或创伤性的情绪？[REWRITE]',
        '[HIGH]悲伤[SEP]谢谢。现在我会问一些问题以了解您的情况。[REWRITE]',
        '[LOW]悲伤[SEP]我知道这可能很困难，但如果您能与我分享这一点，我会非常感激。请问您是否对某人表达或感受到了以下强烈的情绪：[REWRITE]',
        '[HIGH]悲伤[SEP]您认为您应该成为别人的救星吗？[REWRITE]',
        '[LOW]悲伤[SEP]我想知道你对你在这个角色中的角色有什么看法。在目前的情况下，您是否觉得自己是一个受害者，想要将自己的负面情绪转移到其他人身上？[REWRITE]',
        '[HIGH]悲伤[SEP]您觉得您在试图控制某人吗？[REWRITE]',
        '[LOW]悲伤[SEP],嗯，我明白了，谢谢你花时间回答所有这些问题，我仍在努力找出最好的帮助方式。你会说当出现问题时，你有把矛头指向自己的倾向吗？[REWRITE]',
        '[HIGH]悲伤[SEP]在之前的对话中，您是否考虑过其他提出的观点？[REWRITE]',
        '[LOW]悲伤[SEP]很抱歉让您感到难过，为了帮助您，我想深入挖掘一下。如果你不介意我问一下，你正在经历一场艰难的个人危机吗？[REWRITE]',
        '[HIGH]快乐[SEP]那很好！让我推荐一个您可以尝试的协议。[REWRITE]',
        '[LOW]所有情绪[SEP]感谢您通过分享您的感受来信任我。从你告诉我的情况来看，我相信你有 {} 的感觉？[REWRITE]',
        '[HIGH]所有情绪[SEP]我很抱歉。请从下面的情绪中选择最能反映您感受的情绪：[REWRITE]',
        '[LOW]所有情绪[SEP]再见,我要感谢您今天与我分享您的想法和感受。我真的很想很快再见到你，所以请你有空的时候过来看看。[REWRITE]',
        '[HIGH]所有情绪[SEP]这是我的建议，请选择您想尝试的协议[REWRITE]',
        '[LOW]所有情绪[SEP]太好了，我很高兴您愿意尝试一下，希望您会发现它很有帮助。请在准备好后仔细阅读协议，如果完成后可以按“继续”，那就太好了。[REWRITE]',
        '[HIGH]所有情绪[SEP]采取此协议后，您感觉更好还是更糟？[REWRITE]',
        '[LOW]所有情绪[SEP]非常好。如您所知，列表中还有更多协议，您是否认为您可能想尝试其他协议，看看情况如何？[REWRITE]',
        '[HIGH]所有情绪[SEP]您想尝试另一种协议吗？ （病人感觉更糟)[REWRITE]']

# 55957
# 55959: with temp=2

##### 7094 dataset #####
# 56758: 13.44 epoch
# 56766: 6.72 epoch
# 56786: 100 epochs
# 56787: 50 epochs
# 56792: 4500 checkpoint
# 56802: 45000 checkpoint
