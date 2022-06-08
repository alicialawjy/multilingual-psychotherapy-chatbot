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
                evaluate_during_training=False  # best model determined by validation set performance
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
    # # Emotion (Twitter) Dataset (First Tune)
    # df_twitter = pd.read_csv('data/emotions/twitter/twitter_emotions_enzh.csv')
    # df_train_twitter, df_test_twitter = train_test_split(df_twitter, test_size=0.2, shuffle=True, random_state=0, stratify=df_twitter['labels'])

    # sentiment-40k Dataset (First Tune)
    df_train_sentiment40k = pd.read_csv('data/emotions/sentiment-40k/sentiment-40k_train.csv')
    df_test_sentiment40k = pd.read_csv('data/emotions/sentiment-40k/sentiment-40k_test.csv')

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

    #   # Begin First Finetune (Twitter)
    #   model = train_model(epoch = 5,
    #                     best_model_dir= 'emotion_classifier/best_model_twitter',
    #                     use_early_stopping = True,
    #                     early_stopping_delta = 0.01,
    #                     early_stopping_metric_minimize = False,
    #                     early_stopping_patience = 5,
    #                     evaluate_during_training_steps = 1000, 
    #                     evaluate_during_training=True,  # best model determined by validation set performance
    #                     output_dir='emotion_classifier/outputs/first-tune-twitter',
    #                     learning_rate=3e-05,
    #                     model_name = "xlm-roberta-base",
    #                     train_df = df_train_twitter[['text','labels']],
    #                     eval_df = df_test_twitter[['text','labels']])

    # Begin First Finetune (Sentiment-40k)
    model = train_model(epoch = 5,
                    best_model_dir= 'emotion_classifier/best_model_sentiment40k',
                    use_early_stopping = True,
                    early_stopping_delta = 0.01,
                    early_stopping_metric_minimize = False,
                    early_stopping_patience = 5,
                    evaluate_during_training_steps = 1000, 
                    evaluate_during_training=True,  
                    output_dir='emotion_classifier/outputs/first-tune-sentiment40k',
                    learning_rate=3e-05,
                    model_name = "xlm-roberta-base",
                    train_df = df_train_sentiment40k[['text','labels']],
                    eval_df = df_test_sentiment40k[['text','labels']])

    # Second Finetune (Hyperparam Tune)
    best_F1 = 0
    best_epoch = 1
    best_lr = 1e-05

    learning_rate = [1e-05, 2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05]
    for epoch in range(1,11):
        for lr in learning_rate:
            model = train_model(epoch = epoch,
                                output_dir = f'emotion_classifier/outputs/second-tune-EP40k/{str(epoch)}/{str(lr)}',
                                learning_rate = lr,
                                model_name = 'emotion_classifier/best_model_sentiment40k',
                                train_df = df_train_EP[['text','labels']],
                                eval_df = df_val_EP[['text','labels']])

            # evaluate each epoch's performance
            print(f'epoch {epoch}, learning rate {lr}')
            F1 = evaluate(model, df_val_EP)
            if F1 > best_F1:
                best_F1 = F1
                best_epoch = epoch
                best_lr = lr
                print(f'best model updated! lr:{best_lr} epoch: {best_epoch}')

    # Load the best model
    model_best = ClassificationModel(model_type="xlmroberta", 
                                    model_name=f'emotion_classifier/outputs/second-tune-EP40k/{str(best_epoch)}/{str(best_lr)}', 
                                    num_labels=4, 
                                    use_cuda=cuda_available)

    # Check if same results obtained
    print('Evaluating best model')
    evaluate(model_best, df_val_EP)

    # Evaluate saved model on test results
    print('Best Model on Held-Out Test Set')
    evaluate(model_best, df_test_EP)

# LOGS:
# Last Run: 52657 for Twitter Finetuning
# 52660 for hyperparm tuning of twitter model on EP
