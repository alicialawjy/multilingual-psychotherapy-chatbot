import pandas as pd
import random
import re
import torch
from simpletransformers.classification import ClassificationModel

##############################################################################
########           E M O T I O N  C L A S S I F I C A T I O N         ########
########             Model in /emotion-classifier-distilled           ########
##############################################################################

# load emotion classifier outside function to avoid loading model each time during fuinction call
with torch.no_grad():
  cuda_available = torch.cuda.is_available()
  # Empathy Classifier
  EMOTION_CLASSIFIER_NAME = 'emotion-classifier-distilled'
  emotion_classifier = ClassificationModel(model_type="xlmroberta", 
                                          model_name=EMOTION_CLASSIFIER_NAME, 
                                          num_labels=4,
                                          use_cuda=cuda_available)

def get_emotion(user_input, language):
    '''
    Classifies and returns the underlying emotion of a text string
    '''
    # emotion classification labels corr to the language
    labels = {
    "中文(ZH)": {0:"悲伤", 1:"愤怒", 2:"快乐", 3:"焦虑"},
    "English(EN)": {0:"Sad", 1:"Angry", 2:"Happy/Content", 3:"Anxious/Scared"}
    }

    y_pred, _ = emotion_classifier.predict([user_input])

    labels_map_in_lang = labels[language]

    return labels_map_in_lang[y_pred[0]]
  
##############################################################################
########             S E N T E N C E  P R O C E S S I N G             ########
########  Functions to extract and clean sentences before displaying  ########
##############################################################################
def get_sentence(dataset, column_name, language):
    '''
    Extracts a sentence at random without replacement.
    - column_name [str]: the intended prompt (i.e. the column name)
    - user_id [int]: user's id

    Returns: 
    - sentence [str]: the selected prompt utterance.
    '''
    df = dataset                    # get the existing datafram for the user
    column = df[column_name]        # extract the relevant column

    # if the column is out of utterances, replenish list
    if not column.dropna().to_list():
        print(f'Sentences for {column_name} are now depleted. Replenishing sentences.')
        read_df = pd.read_csv(f'utterances/{language}.csv') #, encoding='ISO-8859-1'
        df[column_name] = read_df[column_name]
        column = df[column_name]

    # select a random utterance from the list
    sentence = random.choice(column.dropna().to_list())

    # remove from the list to prevent calling the same ones
    df[column_name] = column[column!=sentence]

    return sentence

def split_sentence(sentence):
    '''
    To make conversations easier to understand, we split each sentence into separate messages using this function.
    - sentence [string]: bot message to be shown to user.

    Returns:
    - the sentences to be outputed in separate segments [tuple]
    '''
    # split by punctuation
    temp_list = re.split('(?<=[.?!]) +', sentence)
    print(temp_list)

    if temp_list[0] == sentence:
        temp_list = re.split('(?<=[。？！：])+', sentence)

    # remove any elements that are empty strings after the split
    if '' in temp_list:
        temp_list.remove('')

    return tuple(temp_list)