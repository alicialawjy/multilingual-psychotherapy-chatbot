# Knowledge Distillation Code adapted from: 
# https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker/blob/master/sagemaker-distillation.ipynb

import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from classify import evaluate

####### K N O W L E D G E  D I S T I L L A T I O N ########
# Overwrite classification arguments 
class Distillation_ClassificationArgs(ClassificationArgs):
    def __init__(self, *args, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

# Overwrite classification model
class Distillation_ClassificationModel(ClassificationModel):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher._move_model_to_device() # place teacher on same device as student
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
        loss_function = nn.CrossEntropyLoss().to(device) 
        loss_logits = (loss_function(student_softmax, teacher_softmax)*self.args.temperature**2) # multiply temp**2 to scale it back.

        # (iii) Cosine Embedding Loss (based on DistilBERT)
        loss_cosine_function = nn.CosineEmbeddingLoss().to(device)
        loss_cosine = loss_cosine_function(teacher_softmax, student_softmax, (torch.ones(teacher_softmax.size()[0])).to(device))

        # Return Loss
        loss = (student_loss + loss_logits + loss_cosine)/3 # Take the average (as per Triple Loss in DistilBERT)
        return (loss, *outputs_student[1:])

########## T R A I N I N G w/ K D ############
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

  # Second Finetuning (EP) 
  student_model = run_training(epoch = 20, 
                              learning_rate = 1e-05,
                              alpha = 0.5,
                              temperature = 6,
                              output_dir = 'distill/2-tune-2-teacher/2nd-tune/1e-05/temp-6/outputs', 
                              best_model_dir = 'distill/2-tune-2-teacher/2nd-tune/1e-05/temp-6/best-model', 
                              student_model_name = 'distill/2-tune-2-teacher/1st-tune/3e-05/best-temp-4', # use best student_model from first tuning
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
