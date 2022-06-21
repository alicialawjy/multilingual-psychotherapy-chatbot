import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader 
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from transformers import Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

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

def train_model(epoch, 
                output_dir,
                learning_rate,
                model_name,
                train_df,
                eval_df,
                best_model_dir = 'outputs/best_model', 
                use_early_stopping=False, 
                early_stopping_delta=0,
                early_stopping_metric = "eval_loss",
                early_stopping_metric_minimize = True,
                early_stopping_patience = 3,
                evaluate_during_training_steps = 2000, 
                evaluate_during_training=False  
                ):

  model_args = ClassificationArgs(num_train_epochs=epoch,           
                                  best_model_dir=best_model_dir,  
                                  use_early_stopping = use_early_stopping,
                                  early_stopping_delta = early_stopping_delta,
                                  early_stopping_metric = early_stopping_metric,
                                  early_stopping_metric_minimize = early_stopping_metric_minimize,
                                  early_stopping_patience = early_stopping_patience,
                                  evaluate_during_training_steps = evaluate_during_training_steps, 
                                  evaluate_during_training=evaluate_during_training,
                                  no_cache=True,                  
                                  save_steps=-1,                  
                                  save_model_every_epoch=False,
                                  output_dir = output_dir,
                                  overwrite_output_dir = True,
                                  learning_rate=learning_rate,    
                                  optimizer='AdamW')            

  model = ClassificationModel(model_type="xlmroberta",  # tried xlmroberta, bert
                            model_name=model_name,      # tried bert-base-chinese, xlm-roberta-base, bert-base-multilingual-cased (mBert), microsoft/infoxlm-base
                            args = model_args,          # see above
                            num_labels=3,               # 3 labels - no empathy, neutral, empathetic
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

# Run finetuning
if __name__ == "__main__":
  ## Datasets
  # EP Empathy Dataset 
  df_train = pd.read_csv('data/empathy/empatheticpersonas/balanced/EN_ZH_train.csv')
  df_val = pd.read_csv('data/empathy/empatheticpersonas/balanced/ZH_val.csv')
  df_test = pd.read_csv('data/empathy/empatheticpersonas/balanced/ZH_test.csv')

  # Use GPU
  GPU = True
  if GPU:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
      device = torch.device("cpu")
  print(f"Using {device}")

  cuda_available = torch.cuda.is_available()

  model = train_model(epoch = 20,
                      learning_rate = 1e-05,
                      model_name = "xlm-roberta-base", 
                      output_dir = 'empathy_classifier/1e05/outputs', 
                      best_model_dir = 'empathy_classifier/1e05/best-model', 
                      use_early_stopping = True,
                      early_stopping_delta = 0.0001,
                      early_stopping_metric = 'mcc',
                      early_stopping_metric_minimize = False,
                      early_stopping_patience = 10,
                      evaluate_during_training_steps = 115, 
                      evaluate_during_training = True,  
                      train_df = df_train[['text','labels']],
                      eval_df = df_val[['text','labels']]
                      )

  # load the best model
  model_best = ClassificationModel(model_type="xlmroberta", 
                                  model_name= 'empathy_classifier/1e05/best-model',
                                  num_labels=3, 
                                  use_cuda=cuda_available)

  # evaluate the best model
  print('Best model with learning rate = 1e-05')
  evaluate(model_best, df_test)

# LOGS:
# 53593: hyperparam with num_labels=3 (checked)
