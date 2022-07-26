import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from simpletransformers.classification import ClassificationModel

def evaluate(model, df_dataset):
  y_pred, _ = model.predict(df_dataset.text.tolist())
  y_true = df_dataset['labels']

  print("Classification Report", classification_report(y_true, y_pred))
  print("Confusion Matrix", confusion_matrix(y_true, y_pred))
  print("F1-Score", f1_score(y_true, y_pred,average='weighted'))
  return f1_score(y_true, y_pred,average='weighted')

def predictions(model, df_dataset):
  y_pred, _ = model.predict(df_dataset.tolist())
  df = pd.DataFrame(zip(df_dataset, y_pred), columns=['rewriting','empathy'])
  print(df['empathy'].value_counts(normalize=True))
  df.to_csv('rewritings_empathy_labelled.csv')

# test data
# df_ECM_test = pd.read_csv('data/emotions/sentiment-40k/sentiment-40k_test.csv')
# df_test_EP = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')
# df_EN = pd.read_csv('data/emotions/EmpatheticPersonas/EN/emotionlabeled_test.csv')
# df_native = pd.read_csv('data/emotions/EmpatheticPersonas/roy_native.csv')
# df_codeswitch = pd.read_csv('data/emotions/EmpatheticPersonas/EP_codeswitch.csv')
# df_empathy_test = pd.read_csv('data/empathy/empatheticpersonas/balanced/ZH_test.csv')
# df_empatheticrewrite = pd.read_csv('data/empathy/EP_empathy_2144_ZH.csv')
df_generated = pd.read_csv('data/rewritings/empathetic_rewritings.csv',index_col=0)

# Model
models = {'empathy': 'empathy_classifier/binary-empathy'}

for checkpt, model_name in models.items():
  cuda_available = torch.cuda.is_available()

  # Load the best model
  model_best = ClassificationModel(model_type="xlmroberta", 
                                  model_name=model_name, 
                                  num_labels=2, # 3 for empathy, 4 for emotion classifier 
                                  use_cuda=cuda_available)

  # print(f'ECM finetuning results for {checkkpt}')
  # evaluate(model_best, df_ECM_test)
  # print(f'{checkpt}')
  # print('ZH Test Set')
  # evaluate(model_best, df_test_EP)
  # print('Native Test Set')
  # evaluate(model_best, df_native)
  # print('EN Test Set')
  # evaluate(model_best, df_EN)
  # print('CodeSwitch Set')
  # evaluate(model_best, df_codeswitch)
  # print(f'Code Switching for Model: {checkpt}')
  predictions(model_best, df_generated['rewriting'])


####### LOGS #########
# ran on job 52784
# 52789 for cleaned en test dataset
# 52812 for native roy 
# sentiment-40k
#   2, 3e-05: 52828
#   5, 3e-05: 52826
# 53012: single best (5 3e-05) results
# 53074: ECM best (5 3e-05)
# 54045: ECM (3e-05 and 1e-05)
# 54066: ECM 9e-06 Batch Size 8
# 54115: ECM 9e-06 Batch Size 32
# 54146: 2nd tune 1e-05
# 54164: 2nd tune 1e-06
# 54262: KD no teacher ECM 1st tuning
# 54363: KD no teacher EP 2nd tuning
# 54388: KD no teacher 805,920,1380,1955 checkpoint models
# 54389: KD 1 teacher 1401, 1725, 2415, 3680, 4945, 6325
# 54401: KD 2 teacher 1st-tuning model 10000, 11750, 12500
# 54405: KD 2 teacher 2nd-tuning model 1265, 1380, 1840
# 54410: 1-tune 1-teacher
# 54411: 1-tune 1-teacher 2nd attempt 2990, 3335
# 54431: 1-tune 0-teacher 1725, 2415

##### HYPERPARAMETER TUNING 2-tune 2-teacher (1st tuning) #####
# 54438: 5e-05 3750 4500 5750
# 54439: 5e-06
# 54445: 3e-05

##### HYPERPARAMETER TUNING 2-tune 2-teacher (2nd tuning) #####
# 54480: 1e-05
# 54482: 2e-05
# 54500: 9e-06 and 3e-05
# 54507+12: 1e-05 temp=3 and 5 evaluate
# 54522: temp=6,7

##### Empathy Classifier #####
# 54554: 2e-05 empathy classifier
# 54564: 9e-06, 2e-05, 3e-05
# 55454: label the other empatheticrewritings

##### Code Switching #####
# 55277: distilled and xlm-r 

