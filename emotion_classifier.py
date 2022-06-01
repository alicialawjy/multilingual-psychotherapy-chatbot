import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader # for the dataloader
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Convert dataframe into dictionary of text and labels
def reader(df):
    texts = df['text'].values.tolist()
    labels = df['label'].values.tolist()

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
      labels.append(b['labels'])

    print(texts)
    print(labels)
    encodings = self.tokenizer(
      texts,                        # what to encode
      return_tensors = 'pt',        # return pytorch tensors
      add_special_tokens = True,    # incld tokens like [SEP], [CLS]
      padding = "max_length",       # pad to max sentence length
      truncation = True)            # truncate if too long
      # max_length= 128              

    encodings['labels'] = torch.tensor(labels)
    return encodings

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    item = {'text': self.texts[idx], 'label': self.labels[idx]}

    return item

# train model
def train_model():
  optimizer = 'AdamW' 
  learning_rate = 3e-05
  epochs = 8
  
  model_args = ClassificationArgs(num_train_epochs=epochs, 
                                        no_save=True, 
                                        no_cache=True, 
                                        overwrite_output_dir=True,
                                        learning_rate=learning_rate,
                                        optimizer=optimizer)

  model = ClassificationModel(model_type="xlmroberta",      # tried xlmroberta, bert
                            model_name="microsoft/infoxlm-base",  # tried bert-base-chinese, xlm-roberta-base, bert-base-multilingual-cased (mBert), microsoft/infoxlm-base
                            args = model_args, 
                            num_labels=4, 
                            use_cuda=cuda_available)
  
  print('Training Model')
  model.train_model(df_train[['text', 'labels']])

  # Validation Set (Internal)
  y_pred, _ = model.predict(df_val.text.tolist())
  y_true = df_val['labels']

  print("Validation Set Classification Report - InfoXLM EN-only")
  # print(f1_score(y_true, y_pred,average='weighted'))
  print(classification_report(y_true, y_pred))
  # print(confusion_matrix(y_true, y_pred))
  
  # # Test Set (Internal)
  # y_pred, _ = model.predict(df_test.text.tolist())
  # y_true = df_test['labels']
  # print("Test Set Classification Report")
  # print(classification_report(y_true, y_pred))
  # print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    GPU = True
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    cuda_available = torch.cuda.is_available()
    df_train = pd.read_csv('data/EN/emotionlabeled_train.csv')
    df_val = pd.read_csv('data/ZH/emotionlabeled_val.csv')
    df_test = pd.read_csv('data/ZH/emotionlabeled_test.csv')

    train_model()