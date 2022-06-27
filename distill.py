# Code adapted from: 
# https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker/blob/master/sagemaker-distillation.ipynb

from torch.utils.data import Dataset 
import pandas as pd
import os
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import classification_report, confusion_matrix, f1_score

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

# Overwrite classification arguments and classification model
class Distillation_ClassificationArgs(ClassificationArgs):
    def __init__(self, *args, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

class Distillation_ClassificationModel(ClassificationModel):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher._move_model_to_device() # Place teacher on same device as student
        self.teacher.model.eval()

    def _calculate_loss(self, model, inputs, loss_fct, num_labels, args): 
        # For more information, see Section 5.2.1 of the report on Training Loss.

        # Compute outputs
        outputs_student = model(**inputs) # student

        with torch.no_grad():
            outputs_teacher = self.teacher.model(**inputs) # teacher

        # Assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Softmax 
        teacher_softmax = F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1) # teacher
        student_softmax = F.softmax(outputs_student.logits / self.args.temperature, dim=-1) # student
        student_logsoftmax = F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1) # student log softmax

        # (i) Classic Supervised Training Loss 
        student_loss = outputs_student.loss

        # (ii) Distillation Loss
        loss_function = nn.CrossEntropyLoss().to(device) # nn.KLDivLoss(reduction="batchmean").to(device) # KLDivLoss: Kullback-Leibler divergence loss // 
        loss_logits = (loss_function(student_softmax, teacher_softmax)*self.args.temperature**2) # multiply temp**2 to scale it back.

        # (iii) Cosine Embedding Loss (based on DistilBERT)
        loss_cosine_function = nn.CosineEmbeddingLoss().to(device)
        loss_cosine = loss_cosine_function(teacher_softmax, student_softmax, (torch.ones(teacher_softmax.size()[0])).to(device))

        # Return Loss
        loss = (student_loss + loss_logits + loss_cosine)/3 # Take the average of the Triple Loss
        return (loss, *outputs_student[1:])

def run_training(epoch, 
                learning_rate,
                output_dir,
                best_model_dir, 
                train_df,
                eval_df,
                student_model_name,
                alpha=0.5,
                temperature=4,
                teacher_model = None,
                use_early_stopping=False, 
                early_stopping_delta=0,
                early_stopping_metric = "eval_loss",
                early_stopping_metric_minimize = True,
                early_stopping_patience = 10,
                evaluate_during_training=False,
                evaluate_during_training_steps = 100, 
                train_batch_size = 8
                ):

  # If there's a teacher model (ie. Knowledge Distillation), use Distillation Class
  if teacher_model:
    print('Teacher Found, running training with Knowledge Distillation.')
    model_args = Distillation_ClassificationArgs(num_train_epochs=epoch,
                                                learning_rate=learning_rate,  
                                                alpha=alpha,
                                                temperature=temperature,
                                                output_dir = output_dir,           
                                                best_model_dir=best_model_dir, 
                                                train_batch_size=train_batch_size,
                                                evaluate_during_training_steps = evaluate_during_training_steps, 
                                                use_early_stopping = use_early_stopping,
                                                early_stopping_delta = early_stopping_delta,
                                                early_stopping_metric = early_stopping_metric,
                                                early_stopping_metric_minimize = early_stopping_metric_minimize,
                                                early_stopping_patience = early_stopping_patience,
                                                evaluate_during_training=evaluate_during_training,
                                                no_cache=True,                  
                                                save_steps=-1,                  
                                                save_model_every_epoch=True,
                                                overwrite_output_dir = True,  
                                                optimizer='AdamW')       

    os.environ["TOKENIZERS_PARALLELISM"] = "false"     

    # Student:
    student_model = Distillation_ClassificationModel(teacher_model = teacher_model,
                                                    model_type="xlmroberta",
                                                    model_name= student_model_name, 
                                                    args = model_args, 
                                                    num_labels=4,  
                                                    use_cuda=cuda_available)

  # If no teacher (i.e: no KD), use normal Classification Model class
  else:
    print('No Teacher Found, running training without Knowledge Distillation.')
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
                                    save_model_every_epoch=True,
                                    output_dir = output_dir,
                                    overwrite_output_dir = True,
                                    learning_rate=learning_rate,    
                                    optimizer='AdamW')  
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Student
    student_model = ClassificationModel(model_type="xlmroberta",
                                        model_name= student_model_name, # 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large'
                                        args = model_args, 
                                        num_labels=4,  
                                        use_cuda=cuda_available)

  student_model.train_model(train_df = train_df,              # training dataset
                            eval_df = eval_df)                # evaluation dataset

  return student_model                  


def evaluate(model, df_dataset):
  y_pred, _ = model.predict(df_dataset.text.tolist())
  y_true = df_dataset['labels']

  print("Classification Report", classification_report(y_true, y_pred))
  print("Confusion Matrix", confusion_matrix(y_true, y_pred))
  print("F1-Score", f1_score(y_true, y_pred,average='weighted'))
  return f1_score(y_true, y_pred,average='weighted')

# Run distillation
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

  # Teacher Models
  first_teacher_model = ClassificationModel(model_type="xlmroberta",
                                            model_name='emotion_classifier/2-tune-ECMxEP/1st-ECM-tune-9e06', # First Model - ECM 1st-tune
                                            num_labels=4,  
                                            use_cuda=cuda_available)

  second_teacher_model = ClassificationModel(model_type="xlmroberta",
                                            model_name='emotion_classifier/2-tune-ECMxEP/2nd-EP-tune-2e05', # Second Model - EP 2nd-tune
                                            num_labels=4,  
                                            use_cuda=cuda_available)

  # Student Models
  # First Finetuning (ECM)
  student_model = run_training(epoch = 20, 
                              learning_rate = 3e-05,
                              temperature = 6,
                              output_dir = 'distill/2-tune-2-teacher/1st-tune/3e-05/temp-6/outputs', 
                              best_model_dir = 'distill/2-tune-2-teacher/1st-tune/3e-05/temp-6/best-model', 
                              student_model_name = 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large',
                              teacher_model = first_teacher_model, 
                              use_early_stopping = True,
                              early_stopping_delta = 0.0001,
                              early_stopping_metric = "mcc",
                              early_stopping_metric_minimize = False,
                              early_stopping_patience = 10,
                              evaluate_during_training=True,
                              evaluate_during_training_steps = 250, 
                              train_batch_size = 8, 
                              train_df = df_train_ECM[['text','labels']],
                              eval_df = df_test_ECM[['text','labels']]
                              )
  
  # load the best model from the first finetuning
  model_best_1st = ClassificationModel(model_type="xlmroberta", 
                                      model_name= 'distill/2-tune-2-teacher/1st-tune/3e-05/temp-6/best-model', 
                                      num_labels=4, 
                                      use_cuda=cuda_available)
  
  print('First Tuning (lr = 3e-05, temp=6) Validation Results')
  evaluate(model_best_1st, df_test_ECM)

  # Second Finetuning (EP) 
  student_model = run_training(epoch = 20, 
                              learning_rate = 1e-05,
                              alpha = 0.5,
                              temperature = 6,
                              output_dir = 'distill/2-tune-2-teacher/2nd-tune/1e-05/temp-6/outputs', 
                              best_model_dir = 'distill/2-tune-2-teacher/2nd-tune/1e-05/temp-6/best-model', 
                              student_model_name = 'distill/2-tune-2-teacher/1st-tune/3e-05/best-temp-4',
                              teacher_model = second_teacher_model, 
                              use_early_stopping = True,
                              early_stopping_delta = 0.0001,
                              early_stopping_metric = "eval_loss",
                              early_stopping_metric_minimize = True,
                              early_stopping_patience = 15,
                              evaluate_during_training=True,
                              evaluate_during_training_steps = 115, 
                              train_batch_size = 8, 
                              train_df = df_train[['text','labels']],
                              eval_df = df_val[['text','labels']]
                              )

  # load the best model
  model_best = ClassificationModel(model_type="xlmroberta", 
                                  model_name= 'distill/2-tune-2-teacher/2nd-tune/1e-05/temp-6/best-model', 
                                  num_labels=4, 
                                  use_cuda=cuda_available)

  # evaluate
  print('Test Performance')
  evaluate(model_best, df_test)

  print('Native Performance')
  evaluate(model_best, df_native)

  print('EN Performance')
  evaluate(model_best, df_EN)

# LOGS:
# 53581: nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large model, 5e-05, 20 epoch,  batch size = 8
#           loss: KLDiv(log_softmax(student),softmax(teacher)) + student_loss
# 53604: new loss = cross_entropy instead of KLDiv and no * (self.args.temperature ** 2)
# 53619: new loss = include cosineembeddingloss
# 53624: KLDiv (remember to use log softmax!) + cosineembeddingloss
# 53632: Using values from DistilBERT - alpha_distil = 5.0; alpha_cos = 1.0; alpha_student = 2.0; temp = 4.0
# 53634: + change to CE
# 53803: Train with augmented data
# 53822: Train with shuffled aug data
# 53834: Augmented data with roy's concatenating method (promising for ZH, bad for EN)
# 53850: Add EN_para data (x good)
# 53902: with syn replace (zh-aug only, no en)
# 53907/ 53918 (larger patience = 20): 2 teacher 2-tune KD w/ best aug dataset (terminated at 2/8 and 6 epochs) very undertuned
# 53908/ 53922 (larger patience = 20): 1 teacher 2-tune KD w/ best aug dataset (terminated at 2 and 6 epochs)
# 53913/ 53924 (using 53922's longer tuned ECM model): 0 teacher 2-tune (no KD) w/ best aug dataset (terminated at 2 and x epochs)

##### RESTARTING WITH NEW MODEL #####
# 54254: 2-tune 0 teachers (1st-tuning)
# 54264/5: 2-tune 0 teachers (2nd-tuning)
# 54358: disable early stop (stop manually) + hide the model_args.json
# 54362: change folder name so it will not take it as a checkpoint!!!
# 54384: 2-tune 1 teacher, using 54254 as the starting model.
# 54395: 2-tune 2 teacher (1st-tuning)
# 54402: 2-tune 2 teacher (2nd-tuning)
# 54408: 1-tune 1 teacher 
# 54409: 08 with eval_loss as early stopping metric
# 54424: 1-tune 0 teacher

##### HYPERPARAMETER TUNING 2-tune 2-teacher (1st tuning) #####
# 54436: 5e-05
# 54437: 5e-06 (cancelled, too slow)
# 54444: 1e-05
# 54442: 3e-05 w/ temp=4 (best)
# 54446: 3e-05 w/ temp=10
# 54447: 3e-05 w/ temp=15
# 54448: 3e-05 w/ temp=5
# 54449: 3e-05 w/ temp=3
# 54451: 3e-05 w/ temp=2
# 54466: 3e-05 w/ temp=6

##### HYPERPARAMETER TUNING 2-tune 2-teacher (2nd tuning - base model 54442 lr=3e-05 and temp=4) #####
# 54475: 1e-05 w/ temp=4
# 54476: 2e-05
# 54483: 9e-06
# 54484: 3e-05
# 54502: 1e-05 w/ temp=5
# 54503: 1e-05 w/ temp=3
# 54515: 1e-05 w/ temp=6
# 54516: temp=7