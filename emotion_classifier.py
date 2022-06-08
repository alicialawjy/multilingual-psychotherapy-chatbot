import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader 
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from transformers import Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Convert dataframe into dictionary of text and labels
def reader(df):
    texts = df['text'].values.tolist()
    labels = df['labels'].values.tolist()

    return {'texts':texts, 'labels':labels}

# DataLoader
class OlidDataset(Dataset):
  def __init__(self, tokenizer, input_set):
    # input_set: dictionary version of the df
    self.texts = input_set['texts']
    self.labels = input_set['labels']
    self.tokenizer = tokenizer

  def collate_fn(self, batch):
    texts = []
    labels = []

    for b in batch:
      texts.append(str(b['text']))
      labels.append(b['label'])

    print(texts)
    print(labels)
    encodings = self.tokenizer(
      texts,                        # what to encode
      return_tensors = 'pt',        # return pytorch tensors
      add_special_tokens = True,    # incld tokens like [SEP], [CLS]
      padding = "max_length",       # pad to max sentence length
      truncation = True,            # truncate if too long
      max_length= 128)              

    encodings['labels'] = torch.tensor(labels)
    return encodings

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    item = {'text': self.texts[idx], 'label': self.labels[idx]}

    return item


def train_model(model_name, epoch, learning_rate, output_dir, train_df, eval_df):
  optimizer = 'AdamW' 
  
  model_args = ClassificationArgs(num_train_epochs=epoch,        # number of epochs
                                  # best_model_dir=best_model_dir,  # directory to save best model
                                  # evaluate_during_training=False,  # best model determined by validation set performance
                                  no_cache=True,                  
                                  save_steps=-1,                  
                                  save_model_every_epoch=False,
                                  output_dir = output_dir,
                                  overwrite_output_dir = True,
                                  learning_rate=learning_rate,    # learning rate
                                  optimizer=optimizer)            # optimizer

  model = ClassificationModel(model_type="xlmroberta",  # tried xlmroberta, bert
                            model_name=model_name,      # tried bert-base-chinese, xlm-roberta-base, bert-base-multilingual-cased (mBert), microsoft/infoxlm-base
                            args = model_args,          # see above
                            num_labels=4,               # 4 labels - sad, happy, fear, anger
                            use_cuda=cuda_available)    # use GPU

  model.train_model(train_df = train_df,                # training dataset
                    eval_df = eval_df)                  # evaluation dataset
                 
  return model


def evaluate(model, df_dataset):
  y_pred, _ = model.predict(df_dataset.text.tolist())
  y_true = df_dataset['labels']

  print("Classification Report", classification_report(y_true, y_pred))
  print("Confusion Matrix", confusion_matrix(y_true, y_pred))
  print("F1-Score", f1_score(y_true, y_pred,average='weighted'))
  return f1_score(y_true, y_pred,average='weighted')

# Run double finetuning
if __name__ == "__main__":
  ## Datasets
  # Emotion (Twitter) Dataset (First Tune)
  df_train_twitter = pd.read_csv('data/emotions/twitter/twitter_emotions_enzh.csv')

  # EmpatheticPersonas (EP) Dataset (Second Tune)
  df_train_EP = pd.read_csv('data/emotions/EmpatheticPersonas/EN-ZH/emotionlabeled_train.csv')
  df_val_EP = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_val.csv')
  df_test_EP = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')

  # Use GPU
  GPU = True
  if GPU:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
      device = torch.device("cpu")
  print(f"Using {device}")

  cuda_available = torch.cuda.is_available()

  # Begin First Finetune 
  model = train_model(model_name = "xlm-roberta-base",
                      epoch = 3,
                      learning_rate = 4e-05,
                      output_dir = 'emotion_classifier/outputs/first-tune-twitter',
                      train_df = df_train_twitter[['text','labels']],
                      eval_df = df_val_EP[['text','labels']])

  # Evaluate FINAL model F1
  F1 = evaluate(model, df_val_EP)

  # Second Finetune (Hyperparam Tune)
  best_F1 = 0
  best_epoch = 1

  for epoch in range(1,10):
    model = train_model(model_name = "emotion_classifier/outputs/first-tune-twitter",
                        output_dir = 'emotion_classifier/outputs/second-tune-EP/'+str(epoch),
                        epoch = epoch,
                        learning_rate=4e-05,
                        train_df = df_train_EP[['text','labels']],
                        eval_df = df_val_EP[['text','labels']])

    # evaluate each epoch's performance
    F1 = evaluate(model, df_val_EP)
    if F1 > best_F1:
      best_F1 = F1
      best_epoch = epoch

  # Load the best model
  model_args = ClassificationArgs(num_train_epochs=best_epoch, 
                                    #best_model_dir = '/content/drive/MyDrive/emotion_classifier/outputs/second-tune-EP',
                                    #evaluate_during_training=True,
                                    no_cache=True, 
                                    save_steps=-1,
                                    save_model_every_epoch=False,
                                    overwrite_output_dir=True,
                                    learning_rate=4e-05,
                                    optimizer='AdamW')

  model_best = ClassificationModel(model_type="xlmroberta",      
                              model_name='emotion_classifier/outputs/second-tune-EP/'+str(best_epoch),  
                              args = model_args, 
                              num_labels=4,
                              use_cuda=cuda_available)

  # check if same results obtained
  print('Evaluating best model')
  evaluate(model_best, df_val_EP)

  # Evaluate saved model on test results
  print('Best Model on Held-Out Test Set')
  evaluate(model_best, df_test_EP)
