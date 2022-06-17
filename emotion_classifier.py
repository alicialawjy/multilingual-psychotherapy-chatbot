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
                train_batch_size = 8,
                use_early_stopping=False, 
                early_stopping_delta=0,
                early_stopping_metric = "eval_loss",
                early_stopping_metric_minimize = True,
                early_stopping_patience = 3,
                evaluate_during_training_steps = 2000, 
                evaluate_during_training=False  # best model determined by validation set performance
                ):

  model_args = ClassificationArgs(num_train_epochs=epoch,  
                                  train_batch_size=train_batch_size,         
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

# Run finetuning
if __name__ == "__main__":
    ## Datasets
    # # Emotion (Twitter) Dataset (First Tune)
    # df_twitter = pd.read_csv('data/emotions/twitter/twitter_emotions_enzh.csv')
    # df_train_twitter, df_test_twitter = train_test_split(df_twitter, test_size=0.2, shuffle=True, random_state=0, stratify=df_twitter['labels'])

    # ECM Dataset (First Tune)
    df_train_ECM = pd.read_csv('data/emotions/sentiment-40k/sentiment-40k_train.csv')
    df_test_ECM = pd.read_csv('data/emotions/sentiment-40k/sentiment-40k_test.csv')

    # EmpatheticPersonas (EP) Dataset (Second Tune)
    df_train_EP = pd.read_csv('data/emotions/EmpatheticPersonas/Augmented/en_zh_concatenating-method.csv')
    df_val_EP = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_val.csv')
    df_test_EP = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')
    df_EN = pd.read_csv('data/emotions/EmpatheticPersonas/EN/emotionlabeled_test.csv')
    df_native = pd.read_csv('data/emotions/EmpatheticPersonas/roy_native.csv')

    # Use GPU
    GPU = True
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    cuda_available = torch.cuda.is_available()

    # First Finetune (Twitter/ ECM)
    # For large datasets, recommended to use epoch 4 and batch size = 64
    model = train_model(epoch = 5, 
                      best_model_dir= 'emotion_classifier/2-tuned-ECM-9e06/batch-64/best-ECM',
                      train_batch_size = 64,
                      use_early_stopping = True,
                      early_stopping_delta = 0.0001,
                      early_stopping_metric = "eval_loss",
                      early_stopping_metric_minimize = True,
                      early_stopping_patience = 5,
                      evaluate_during_training_steps = 40, 
                      evaluate_during_training= True,  
                      output_dir= 'emotion_classifier/2-tuned-ECM-9e06/batch-64/outputs',
                      learning_rate= 9e-06,
                      model_name = "xlm-roberta-base",
                      train_df = df_train_ECM[['text','labels']],
                      eval_df = df_test_ECM[['text','labels']])

    # # Second Finetune (EP - Hyperparam Tune)
    # best_F1 = 0
    # learning_rate = [2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05]
    # for lr in learning_rate:
    #   model = train_model(epoch = 20,
    #                       learning_rate = lr,
    #                       best_model_dir= f'emotion_classifier/2-tuned-ECM-1e05/2nd-tuning/{str(lr)}/best-final',
    #                       output_dir = f'emotion_classifier/2-tuned-ECM-1e05/2nd-tuning/{str(lr)}/outputs', 
    #                       use_early_stopping = True,
    #                       early_stopping_delta = 0.0001,
    #                       early_stopping_metric = "eval_loss",
    #                       early_stopping_metric_minimize = True,
    #                       early_stopping_patience = 10,
    #                       evaluate_during_training_steps = 50, 
    #                       evaluate_during_training= True, 
    #                       model_name = 'emotion_classifier/2-tuned-ECM-1e05/1st-tuning/best-ECM', 
    #                       train_df = df_train_EP[['text','labels']],
    #                       eval_df = df_test_EP[['text','labels']])
      
    #   # load the best model for this epoch
    #   best_model = ClassificationModel(model_type="xlmroberta", 
    #                                   model_name= f'emotion_classifier/2-tuned-ECM-1e05/2nd-tuning/{str(lr)}/best-final', 
    #                                   num_labels=4, 
    #                                   use_cuda=cuda_available)

    #   # evaluate its performance
    #   print(f'ECM + EP (2-tuned) with learning rate {lr}')
      
    #   # Test Result
    #   print('Held-Out Test Set')
    #   evaluate(best_model, df_test_EP)

    #   # Native Result
    #   print('Native Test Set')
    #   evaluate(best_model, df_native)

    #   # EN Result
    #   print('EN Test Set')
    #   evaluate(best_model, df_EN)

# LOGS:
# Last Run: 52657 for Twitter Finetuning
# 52660 for hyperparm tuning of twitter model on EP
# 52666 for sentiment40k finetuning + hyperparam tune on EP
# 52676 for single hyperparm tune on EP
# 53981: twitter hypertune (done with twitter)
# 53982 (too high patience + delta): ECM hypertune
# 54034: Tune with eval_loss instead of mcc. 
#       ECM (lr=5e-05 STOPPED, TOO LARGE, epoch=5, patience=5, delta = 0.0001) and EP (epoch=20, patience=10, delta=0.0001)
# 54035: 34 with ECM(lr=3e-05, promising)
# 54036: 34 with ECM(lr=1e-05, smaller better!)
# 54048: 34 with ECM(lr=5e-06, )
# 54050: 34 with ECM(lr=2e-05 )
# 54061: 34 with ECM(lr=8e-06 )
# 54062: 34 with ECM(lr=9e-06, batch size = 8, BEST)
# 54074: 62 with batch size = 16, eval_steps = 150
# 54076: 62 with batch size = 32, eval_steps = 80
# 54077: 62 with batch size = 64, eval_steps = 40
# 54082: 62 with batch size = 128, eval_steps = 20