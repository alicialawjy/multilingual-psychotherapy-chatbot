from unittest.util import _MAX_LENGTH
from tqdm import tqdm
import pandas as pd
#from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments, BertPreTrainedModel, BertModel, BertTokenizer
import torch
import torch.nn as nn
from dont_patronize_me import DontPatronizeMe
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from simpletransformers.classification import ClassificationModel, ClassificationArgs, MultiLabelClassificationModel, MultiLabelClassificationArgs
from urllib import request
import pandas as pd
import logging
import torch
from collections import Counter
from ast import literal_eval  
from sklearn.metrics import f1_score

torch.cuda.empty_cache()

# Convert dataframe into dictionary of text and labels
def reader(df):
  texts = df['text'].values.tolist()
  labels = df['label'].values.tolist()

  return {'texts': texts, 'labels': labels}

# Data Loader
class OlidDataset(Dataset):
  def __init__(self, tokenizer, input_set):
    self.texts = input_set['texts']
    self.labels = input_set['labels']
    self.tokenizer = tokenizer

  def collate_fn(self, batch):
    texts = []
    labels = []

    for b in batch:
      texts.append(str(b['text']))
      labels.append(b['label'])

    encodings = self.tokenizer(
      texts,
      return_tensors = 'pt',
      add_special_tokens = True,
      padding = True,
      truncation = True,
      max_length= 128
      )

    encodings['labels'] = torch.tensor(labels)
    return encodings

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    item = {'text': self.texts[idx], 'label': self.labels[idx]}

    return item

def labels2file(p, outf_path):
    with open(outf_path,'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi])+'\n')

def train_model():
    optimizer = 'AdamW'
    learning_rate = 2e-05
    epochs = 4
    
    model_args = ClassificationArgs(num_train_epochs=epochs, 
                                          no_save=True, 
                                          no_cache=True, 
                                          overwrite_output_dir=True,
                                          learning_rate=learning_rate,
                                          optimizer=optimizer)

    model = ClassificationModel("roberta", 
                              "roberta-base", 
                              args = model_args, 
                              num_labels=2, 
                              use_cuda=cuda_available)
    
    model.train_model(df_train[['text', 'label']])

#     Validation Set (Internal)
    y_pred, _ = model.predict(df_val.text.tolist())
    y_true = df_val['label']
    print("Classification metrics for validation set")
    print(classification_report(y_true, y_pred))
    
#     Test Set (Internal)
    y_pred, _ = model.predict(df_test.text.tolist())
    print("Classification metrics for test set(internal)")
    print(classification_report(df_test['label'],   y_pred))
    
    # Test Set (External)
    y_pred, _ = model.predict(df_submission.text.tolist()) 
    labels2file([[k] for k in y_pred], 'task1.txt')



if __name__ == "__main__":
  # Fix Device
  GPU = True
  if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    device = torch.device("cpu")
  print(f"Using {device}")

  cuda_available = torch.cuda.is_available()

  df_train = pd.read_csv('datasets/training_data/data_augmentation/df_updown_paraphrased.csv', index_col=0)
  df_val = pd.read_csv('datasets/df_val.csv', index_col=0)
  df_test = pd.read_csv('datasets/df_test.csv', index_col=0)
    
  col_names = df_test.columns

  df_submission = pd.read_csv("datasets/task4_test.tsv", sep='\t', index_col = 0, header = None, names = col_names)

  train_model()
  


