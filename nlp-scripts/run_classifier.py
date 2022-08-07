'''
Script used to train the 3 classification models: (i) emotion, (ii) empathy and (iii) semantic.
The below code is the code for the double finetuned emotion classifier.
- To train the empathy and semantic classifier, replace train_df, eval_df and num_labels in train_model()

For Knowledge Distillation, please see run_distillation.py
For evaluating the classification models, see classify.py
'''
import torch
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from classify import evaluate

def train_model(epoch, 
                learning_rate,
                output_dir,
                best_model_dir,
                model_name,
                train_df,
                eval_df,
                num_labels,
                train_batch_size = 8,
                use_early_stopping=False, 
                early_stopping_delta=0,
                early_stopping_metric = "eval_loss",
                early_stopping_metric_minimize = True,
                early_stopping_patience = 3,
                evaluate_during_training_steps = 2000, 
                evaluate_during_training=False
                ):

  model_args = ClassificationArgs(num_train_epochs=epoch,     
                                  learning_rate=learning_rate,                       
                                  train_batch_size=train_batch_size,   
                                  # early stopping parameters
                                  use_early_stopping = use_early_stopping,
                                  early_stopping_delta = early_stopping_delta,
                                  early_stopping_metric = early_stopping_metric,
                                  early_stopping_metric_minimize = early_stopping_metric_minimize,
                                  early_stopping_patience = early_stopping_patience,
                                  evaluate_during_training_steps = evaluate_during_training_steps, 
                                  evaluate_during_training=evaluate_during_training,
                                  # model saving parameters          
                                  best_model_dir=best_model_dir,  
                                  output_dir = output_dir,
                                  no_cache=True,                  
                                  save_steps=-1,                  
                                  save_model_every_epoch=False,
                                  overwrite_output_dir = True
                                  )            

  model = ClassificationModel(model_type="xlmroberta",  
                            model_name=model_name,      
                            args = model_args,          
                            num_labels=num_labels,               # 4 labels (sad, happy, fear, anger) for emotion classification
                            use_cuda=cuda_available)    # use GPU

  model.train_model(train_df = train_df,                # training dataset
                    eval_df = eval_df)                  # evaluation dataset
                 
  return model


# Run finetuning
if __name__ == "__main__":
  ## Datasets
  # ECM Dataset (First Tune)
  df_train_ECM = pd.read_csv('data/emotions/ECM/ECM_train.csv')
  df_test_ECM = pd.read_csv('data/emotions/ECM/ECM_test.csv')

  # EmpatheticPersonas Dataset (Second Tune)
  df_train = pd.read_csv('data/emotions/EmpatheticPersonas/EP_train_augmented.csv') 
  df_train = df_train.sample(frac=1).reset_index(drop=True) # shuffle the dataset
  df_val = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_val.csv')
  df_test = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')
  df_EN = pd.read_csv('data/emotions/EmpatheticPersonas/EN/emotionlabeled_test.csv')
  df_native = pd.read_csv('data/emotions/EmpatheticPersonas/EP_native.csv')

  # Use GPU
  GPU = True
  if GPU:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
      device = torch.device("cpu")
  print(f"Using {device}")

  cuda_available = torch.cuda.is_available()

  # First Finetune (ECM)
  model = train_model(epoch = 20, 
                      learning_rate= 9e-06,
                      num_labels = 4,
                      model_name = "xlm-roberta-base",
                      output_dir= 'emotion_classifier/1st-tuning/outputs',
                      best_model_dir= 'emotion_classifier/1st-tuning/best-model',
                      train_df = df_train_ECM[['text','labels']],
                      eval_df = df_test_ECM[['text','labels']],
                      train_batch_size = 8,
                      use_early_stopping = True,
                      early_stopping_delta = 0.0001,
                      early_stopping_metric = "eval_loss",
                      early_stopping_metric_minimize = True,
                      early_stopping_patience = 10,
                      evaluate_during_training_steps = 250, 
                      evaluate_during_training= True)

  # Second Finetune (EP)
  # (comment this section if double finetuning not required)
  model = train_model(epoch = 20, 
                      learning_rate = 2e-05, 
                      num_labels = 4,
                      model_name = 'emotion_classifier/1st-tuning/best-model',      # use the best model from the first tuning
                      best_model_dir = 'emotion_classifier/2nd-tuning/best-model', 
                      output_dir = 'emotion_classifier/2nd-tuning/outputs', 
                      use_early_stopping = True, 
                      early_stopping_delta = 0.0001, 
                      early_stopping_metric = "eval_loss", 
                      early_stopping_metric_minimize = True, 
                      early_stopping_patience = 10, 
                      evaluate_during_training= True, 
                      evaluate_during_training_steps = 115, 
                      train_batch_size = 8, 
                      train_df = df_train[['text','labels']], 
                      eval_df = df_test[['text','labels']])

  # load the best model for this epoch
  best_model = ClassificationModel(model_type="xlmroberta", 
                                  model_name= 'emotion_classifier/2nd-tuning/best-model', 
                                  num_labels=4, 
                                  use_cuda=cuda_available)

  # Evaluate
  # Test Result
  print('Held-Out Test Set')
  evaluate(best_model, df_test)

  # Native Result
  print('Native Test Set')
  evaluate(best_model, df_native)

  # EN Result
  print('EN Test Set')
  evaluate(best_model, df_EN)
