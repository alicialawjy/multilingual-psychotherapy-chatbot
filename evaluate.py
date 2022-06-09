import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from simpletransformers.classification import ClassificationModel

def evaluate(model, df_dataset):
  y_pred, _ = model.predict(df_dataset.text.tolist())
  y_true = df_dataset['labels']

  print(df_dataset.text.tolist()[y_pred!=y_true])

  print("Classification Report", classification_report(y_true, y_pred))
  print("Confusion Matrix", confusion_matrix(y_true, y_pred))
  print("F1-Score", f1_score(y_true, y_pred,average='weighted'))
  return f1_score(y_true, y_pred,average='weighted')

# test data
# df_EP_ZH_val = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_val.csv')
# df_EP_ZH = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')
# df_EP_EN = pd.read_csv('data/emotions/EmpatheticPersonas/EN/emotionlabeled_test.csv')
df_ZH_native = pd.read_csv('data/emotions/Native Dataset/roy_native.csv') # ('data/emotions/NLPCC2014/NLPCC2014(ZH-Native).csv')
# df_codeswitch = pd.read_csv('data/emotions/EmpatheticPersonas/EP_codeswitch.csv')

# fill in with the best params 
models = {'twitter': 'emotion_classifier/outputs/second-tune-EP/10/3e-05'} # twitter best
# 'single': 'emotion_classifier/outputs/single-tune/5/3e-05', # single tune best
# 'sentiment40k': 'emotion_classifier/outputs/second-tune-EP40k/6/6e-05', # sentiment-40k best

for (ft, model_name) in models.items():
    cuda_available = torch.cuda.is_available()

    # Load the best model
    model_best = ClassificationModel(model_type="xlmroberta", 
                                    model_name=model_name, 
                                    num_labels=4, 
                                    use_cuda=cuda_available)
    
    print(f'Results for {ft} finetuning')

    # # 1: Sanity Check
    # print('Sanity Check on Validation Set')
    # evaluate(model_best, df_EP_ZH_val)

    # # 2: Test (ZH) Performance
    # print('ZH Test Set')
    # evaluate(model_best, df_EP_ZH)

    # # 3: Test (EN) Performance
    # print('EN Test Set')
    # evaluate(model_best, df_EP_EN)

    # 4: Native ZH Performance
    print('ZH Native Set')
    evaluate(model_best, df_ZH_native)

    # # 5: Code Switch Performance
    # print('CodeSwitch Set')
    # evaluate(model_best, df_codeswitch)

# ran on job 52784
# 52789 for cleaned en test dataset
# 52812 for native roy 