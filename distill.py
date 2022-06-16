## Code extracted from: https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker/blob/master/sagemaker-distillation.ipynb
from torch.utils.data import Dataset 
import pandas as pd
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
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

class Distillation_ClassificationModel(ClassificationModel):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self.teacher._move_model_to_device()
        self.teacher.model.eval()

    def _calculate_loss(self, model, inputs, loss_fct, num_labels, args): 
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

        # (i) Student Loss (Typical Classification Loss)
        student_loss = outputs_student.loss

        # (ii) Teacher-Student Loss
        loss_function = nn.CrossEntropyLoss().to(device) # nn.KLDivLoss(reduction="batchmean").to(device) # KLDivLoss: Kullback-Leibler divergence loss // 
        loss_logits = (loss_function(student_softmax, teacher_softmax)*self.args.temperature**2) # multiply temp**2 to scale it back.

        # (iii) Cosine Loss (based on DistilBERT)
        loss_cosine_function = nn.CosineEmbeddingLoss().to(device)
        loss_cosine = loss_cosine_function(teacher_softmax, student_softmax, (torch.ones(teacher_softmax.size()[0])).to(device))

        # Return Loss
        loss = (student_loss + loss_logits + loss_cosine)/3 # self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, *outputs_student[1:])

def run_training(student_model_name,
                epoch, 
                output_dir,
                learning_rate,
                train_df,
                eval_df,
                teacher_model = None,
                best_model_dir = 'outputs/best_model', 
                use_early_stopping=False, 
                early_stopping_delta=0,
                early_stopping_metric = "eval_loss",
                early_stopping_metric_minimize = True,
                early_stopping_patience = 3,
                evaluate_during_training_steps = 2000, 
                evaluate_during_training=False  
                ):

  # if there's a teacher model (ie. Knowledge Distillation), use Distillation Class
  if teacher_model:
    print('Teacher Found, running training with Knowledge Distillation.')
    model_args = Distillation_ClassificationArgs(alpha=0.5,
                                                temperature=2.0,
                                                train_batch_size=8, # batch size 32
                                                num_train_epochs=epoch,           
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

    # Student:
    student_model = Distillation_ClassificationModel(teacher_model = teacher_model,
                                                    model_type="xlmroberta",
                                                    model_name= student_model_name, # 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large'
                                                    args = model_args, 
                                                    num_labels=4,  
                                                    use_cuda=cuda_available)

  # if no teacher (i.e: no KD), use normal Classification Model class
  else:
    print('No Teacher Found, running training without Knowledge Distillation.')
    model_args = Distillation_ClassificationArgs(num_train_epochs=epoch,           
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

    # Student
    student_model = ClassificationModel(model_type="xlmroberta",
                                        model_name= student_model_name, # 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large'
                                        args = model_args, 
                                        num_labels=4,  
                                        use_cuda=cuda_available)

  student_model.train_model(train_df = train_df,              # training dataset
                          eval_df = eval_df)                  # evaluation dataset

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
    # sentiment-40k Dataset (First Tune)
    df_train_sentiment40k = pd.read_csv('data/emotions/sentiment-40k/sentiment-40k_train.csv')
    df_test_sentiment40k = pd.read_csv('data/emotions/sentiment-40k/sentiment-40k_test.csv')

    # EmpatheticPersonas (EP) Dataset (Second Tune)
    df_train = pd.read_csv('data/emotions/EmpatheticPersonas/Augmented/en_zh_concatenating-method.csv') #'data/emotions/EmpatheticPersonas/EN-ZH/emotionlabeled_train.csv')
    df_train = df_train.sample(frac=1).reset_index(drop=True) # shuffle the dataset
    df_val = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_val.csv')
    df_test = pd.read_csv('data/emotions/EmpatheticPersonas/ZH/emotionlabeled_test.csv')
    df_EN = pd.read_csv('data/emotions/EmpatheticPersonas/EN/emotionlabeled_test.csv')
    df_native = pd.read_csv('data/emotions/Native Dataset/roy_native.csv')

    # Use GPU
    GPU = True
    if GPU:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")

    cuda_available = torch.cuda.is_available()

    # Begin First Finetune (Sentiment-40k) w/ teacher
    first_teacher_model = ClassificationModel(model_type="xlmroberta",
                                              model_name='emotion_classifier/best_model_sentiment40k', #'saved_models/2-tuned 5epoch 3e-05lr',
                                              num_labels=4,  
                                              use_cuda=cuda_available)

    # Begin First Finetune (Sentiment-40k) 
    student_model = run_training(teacher_model = first_teacher_model,
                                student_model_name = 'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large',
                                epoch = 20, #epoch,
                                learning_rate = 5e-05, #lr,
                                output_dir = 'distillation/2-tune-2-teachers/1st-tune/outputs', #f'empathy_classifier/outputs/{str(epoch)}/{str(lr)}', 
                                best_model_dir = 'distillation/2-tune-2-teachers/1st-tune-best', #f'empathy_classifier/best_model/{str(epoch)}/{str(lr)}', 
                                use_early_stopping = True,
                                early_stopping_delta = 0.01,
                                early_stopping_metric_minimize = False,
                                early_stopping_patience = 20,
                                evaluate_during_training_steps = 500, 
                                evaluate_during_training=True,
                                train_df = df_train_sentiment40k[['text','labels']],
                                eval_df = df_test_sentiment40k[['text','labels']]
                                 )

    # Second Finetune (EP) w/ teacher
    second_teacher_model = ClassificationModel(model_type="xlmroberta",
                                              model_name='emotion_classifier/outputs/second-tune-EP40k/5/3e-05', #'saved_models/2-tuned 5epoch 3e-05lr',
                                              num_labels=4,  
                                              use_cuda=cuda_available)

    student_model = run_training(teacher_model = second_teacher_model,
                                student_model_name = 'distillation/2-tune-2-teachers/1st-tune-best',
                                epoch = 20, #epoch,
                                learning_rate = 5e-05, #lr,
                                output_dir = 'distillation/2-tune-2-teachers/2nd-tune/outputs', #f'empathy_classifier/outputs/{str(epoch)}/{str(lr)}', 
                                best_model_dir = 'distillation/2-tune-2-teachers/2nd-tune-best', #f'empathy_classifier/best_model/{str(epoch)}/{str(lr)}', 
                                use_early_stopping = True,
                                early_stopping_delta = 0.01,
                                early_stopping_metric_minimize = False,
                                early_stopping_patience = 20,
                                evaluate_during_training_steps = 500, 
                                evaluate_during_training=True,
                                train_df = df_train[['text','labels']],
                                eval_df = df_val[['text','labels']]
                                )

    # load the best model
    model_best = ClassificationModel(model_type="xlmroberta", 
                                    model_name= 'distillation/2-tune-2-teachers/2nd-tune-best', 
                                    num_labels=4, 
                                    use_cuda=cuda_available)

    # evaluate
    print('KD w/ 2Tuned 2 Teacher - Validation Performance')
    evaluate(model_best, df_val)

    print('KD w/ 2Tuned 2 Teacher -  Test Performance')
    evaluate(model_best, df_test)

    print('KD w/ 2Tuned 2 Teacher -  Native Performance')
    evaluate(model_best, df_native)

    print('KD w/ 2Tuned 2 Teacher -  EN Performance')
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
# 53907: 2 teacher 2-tune KD w/ best aug dataset (terminated at 2 and 6 epochs) very undertuned
# 53908/ 53917 (larger patience = 20): 1 teacher 2-tune KD w/ best aug dataset (terminated at 2 and 6 epochs)
# 53913: 0 teacher 2-tune (no KD) w/ best aug dataset (terminated at 2 and x epochs)
