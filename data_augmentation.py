import pandas as pd
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import nltk
import re
import random
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords, wordnet


# Fix Device
GPU = True
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f"Using {device}")

'''
PART ONE: Paraphrasing using Pegasus
'''
MODEL_NAME = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# Tokenise, truncate and pad
def get_response(input_text, num_return_sequences):
    batch = tokenizer.prepare_seq2seq_batch([input_text],
                                            truncation=True,
                                            padding='longest',
                                            return_tensors="pt").to(device)
    translated = model.generate(**batch,
                                num_beams=num_return_sequences,
                                num_return_sequences=num_return_sequences,
                                temperature=1.5).to(device)

    return tokenizer.batch_decode(translated, skip_special_tokens=True)

# Generate paraphrased sentences
def main_para(train_df):
    train = train_df
    train['text'] = train['text'].apply(get_response, num_return_sequences=3)
    generated = train.explode('text')
    generated = generated.dropna()
    generated = generated.drop_duplicates()
    return generated

'''
PART TWO: Synonym Replacement
'''
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if word != l.name():
                synonyms.append(l.name())
    return synonyms

def find_synonym(text):
    sentences = [text]
    stop_words = stopwords.words('english')
    texts = text.split()
    texts = texts[:len(texts) // 2]
    #  Exclude stop words, numeric text and any uppercased words (as these are most likely names/ places)
    word_bag = [i for i in texts if i not in stop_words
                and any(map(str.isupper, i)) is False
                and not i.isnumeric() and i.isalpha()] 
    # Replace only every alternate word
    word_bag = [word_bag[i] for i in range(len(word_bag)) if i % 2 != 0]
    try:
        for word in word_bag:
            similar_words = get_synonyms(word)
            if similar_words is not None:
                similar_words = [re.sub('[^A-Za-z ]+', ' ', sent) for sent in similar_words]
                similar = random.choice(similar_words)
                text = re.sub(word, similar, text)
    except Exception as e:
        e # print(e)
    return text

# Run synonym replacement
def main_synonym(train_df):
    train = train_df
    train['text'] = train['text'].apply(find_synonym)
    return train

'''
Main Code
'''
if __name__ == "__main__":
    # Dataset
    df_train = pd.read_csv('data/emotions/EmpatheticPersonas/EN/emotionlabeled_train.csv')

    # Paraphrasing only
    paraphrased = main_para (df_train) # dataframe
    paraphrased.to_csv('data/emotions/EmpatheticPersonas/Augmented/paraphrased.csv')

    # Synonym Replacement on Paraphrased Text
    syn_and_para = main_synonym(paraphrased)
    syn_and_para.to_csv('data/emotions/EmpatheticPersonas/Augmented/syn_and_para.csv')


    


