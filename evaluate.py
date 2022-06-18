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

# test data
# df_ECM_test = pd.read_csv('data/emotions/sentiment-40k/sentiment-40k_test.csv')
df_test_EP = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')
df_EN = pd.read_csv('data/emotions/EmpatheticPersonas/EN/emotionlabeled_test.csv')
df_native = pd.read_csv('data/emotions/EmpatheticPersonas/roy_native.csv')

# models we want to test
models = {'1e-05': 'emotion_classifier/2-tuned-ECM-9e06/2nd-tuning-1e05/outputs/checkpoint-1265'}
#,
#          '1e-05': 'emotion_classifier/2-tuned-ECM-1e05/1st-tuning/best-ECM'}

for lr,model_name in models.items():
  cuda_available = torch.cuda.is_available()

  # Load the best model
  model_best = ClassificationModel(model_type="xlmroberta", 
                                  model_name=model_name, 
                                  num_labels=4, 
                                  use_cuda=cuda_available)

  print(f'ECM finetuning results for learning rate = {lr}')
  print('ZH Test Set')
  evaluate(model_best, df_test_EP)
  print('Native Test Set')
  evaluate(model_best, df_native)
  print('EN Test Set')
  evaluate(model_best, df_EN)
  


# # df_EP_ZH_val = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_val.csv')
# df_EP_ZH = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')
# df_EP_EN = pd.read_csv('data/emotions/EmpatheticPersonas/EN/emotionlabeled_test.csv')
# df_ZH_native = pd.read_csv('data/emotions/Native Dataset/roy_native.csv') # ('data/emotions/NLPCC2014/NLPCC2014(ZH-Native).csv')
# df_codeswitch = pd.read_csv('data/emotions/EmpatheticPersonas/EP_codeswitch.csv')

# fill in with the best params 
# models = {'sentiment40k': 'emotion_classifier/outputs/second-tune-EP40k/5/3e-05'} # sentiment-40k best
# 'single': 'emotion_classifier/outputs/single-tune/5/3e-05',
# 'sentiment40k': 'emotion_classifier/outputs/second-tune-EP40k/2/3e-05'
# 'twitter': 'emotion_classifier/outputs/second-tune-EP/3/4e-05' 

# for (ft, model_name) in models.items():
#     cuda_available = torch.cuda.is_available()

#     # Load the best model
#     model_best = ClassificationModel(model_type="xlmroberta", 
#                                     model_name=model_name, 
#                                     num_labels=4, 
#                                     use_cuda=cuda_available)
    
#     print(f'Results for {ft} finetuning')

#     # # 1: Sanity Check
#     # print('Sanity Check on Validation Set')
#     # evaluate(model_best, df_EP_ZH_val)

#     # 2: Test (ZH) Performance
#     print('ZH Test Set')
#     evaluate(model_best, df_EP_ZH)

#     # 3: Test (EN) Performance
#     print('EN Test Set')
#     evaluate(model_best, df_EP_EN)

#     # 4: Native ZH Performance
#     print('ZH Native Set')
#     evaluate(model_best, df_ZH_native)

#     # 5: Code Switch Performance
#     print('CodeSwitch Set')
#     evaluate(model_best, df_codeswitch)

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
# 54144: 2nd tune 1e-06 