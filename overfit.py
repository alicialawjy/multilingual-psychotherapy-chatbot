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
# df_EP_ZH_train = pd.read_csv('data/emotions/EmpatheticPersonas/EN-ZH/emotionlabeled_train.csv')
df_EP_ZH_val = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_val.csv')
df_EP_ZH_test = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')

# check the test performance
models = {'single': 'emotion_classifier/outputs/single-tune/', # single tune best
'sentiment40k': 'emotion_classifier/outputs/second-tune-EP40k/', # sentiment-40k best
'twitter': 'emotion_classifier/outputs/second-tune-EP/'} # twitter best
cuda_available = torch.cuda.is_available()
learning_rate = [1e-05, 2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05]
test = {}
val = {}
for (ft, model_name) in models.items():
    ft_epoch_test= []
    ft_epoch_val= []
    for epoch in range(1,11):
        ft_lr_test=[]
        ft_lr_val=[]
        for lr in learning_rate:
            # Load the model
            model = ClassificationModel(model_type="xlmroberta", 
                                            model_name=model_name+f'{str(epoch)}/{str(lr)}', 
                                            num_labels=4, 
                                            use_cuda=cuda_available)

            # 1: Validation (ZH) Performance
            print('ZH Validation Set')
            F1_val = evaluate(model, df_EP_ZH_val)
            ft_lr_val.append(F1_val)

            # 2: Train (EN-ZH) Performance
            print('EN-ZH Test Set')
            F1_test = evaluate(model, df_EP_ZH_test)
            ft_lr_test.append(F1_test)

        ft_epoch_test.append(ft_lr_test)
        ft_epoch_val.append(ft_lr_val)
    test[ft]=ft_epoch_test
    val[ft]=ft_epoch_val

# print results
print('Validation and Test Scores')
print(test)
print(val)

# Slurm 52717: test-val
# 52725: train-val
