'''
Script used to train classification models.
'''

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from simpletransformers.classification import ClassificationModel

def evaluate(model, df_dataset):
  '''
  Function used to evaluate model performance. 
  Prints classification report, confusion matrix and F1-score.
  - model [ClassificationModel object]: model to be evaluated 
  - df_dataset [DataFrame object]: dataset with columns 'text' (the input) and 'labels' (corr. true labels)

  Returns: F1 score
  '''
  y_pred, _ = model.predict(df_dataset.text.tolist())
  y_true = df_dataset['labels']

  print("Classification Report", classification_report(y_true, y_pred))
  print("Confusion Matrix", confusion_matrix(y_true, y_pred))
  print("F1-Score", f1_score(y_true, y_pred, average='weighted'))
  return f1_score(y_true, y_pred,average='weighted')

def predictions(model, input_list, filename):
  '''
  Function used to append and export model predictions.
  - model [ClassificationModel object]: model used to provide predictions
  - input_list [list]: list of the inputs to be predicted
  - filename [string]: filename to export predictions (if any)
  '''
  y_pred, _ = model.predict(input_list)
  df = pd.DataFrame(zip(input_list, y_pred), columns=['text','labels'])
  # print(df['labels'].value_counts(normalize=True))
  if filename:
    df.to_csv(filename)


if __name__ == "__main__":
  # Test data
  df_test = pd.read_csv('data/rewritings/empathetic_rewritings.csv',index_col=0)

  # Use GPU
  cuda_available = torch.cuda.is_available()

  # Load the model
  CLASSIFIER_NAME = "empathy_classifier/binary-empathy"           # give relative path of model directory where classifier is stored
  model_best = ClassificationModel(model_type = "xlmroberta", 
                                  model_name = CLASSIFIER_NAME, 
                                  num_labels = 2,                 # 2 for empathy, 4 for emotion classifier 
                                  use_cuda = cuda_available)

  predictions(model_best, df_test['rewriting'].tolist(), filename='rewritings_empathy_labelled.csv')


# 57886: evaluate our rewritings (89.68% empathetic responses)