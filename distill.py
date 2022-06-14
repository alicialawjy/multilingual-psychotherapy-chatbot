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

        # (i) Student Loss (Typical Classification Loss)
        student_loss = outputs_student.loss

        # (ii) Teacher-Student Loss
        loss_function = nn.CrossEntropyLoss().to(device) # KLDivLoss(reduction="batchmean") # KLDivLoss: Kullback-Leibler divergence loss
        loss_logits = loss_function(student_softmax, teacher_softmax)

        # (iii) Cosine Loss
        loss_cosine_function = nn.CosineEmbeddingLoss().to(device)
        loss_cosine = loss_cosine_function(teacher_softmax, student_softmax, (torch.ones(teacher_softmax.size()[0])).to(device))

        # Return Average Loss
        loss = (student_loss + loss_logits + loss_cosine)/3 # self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, *outputs_student[1:])

def run_training(epoch, 
                output_dir,
                learning_rate,
                train_df,
                eval_df,
                best_model_dir = 'outputs/best_model', 
                use_early_stopping=False, 
                early_stopping_delta=0,
                early_stopping_metric = "eval_loss",
                early_stopping_metric_minimize = True,
                early_stopping_patience = 3,
                evaluate_during_training_steps = 2000, 
                evaluate_during_training=False  
                ):

    model_args = Distillation_ClassificationArgs(alpha=0.5,
                                                temperature=4.0,
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

    # Teacher: finetuned XLM-R model
    teacher_model = ClassificationModel(model_type="xlmroberta",
                                    model_name='emotion_classifier/outputs/second-tune-EP40k/5/3e-05', #'saved_models/2-tuned 5epoch 3e-05lr', 
                                    args = model_args, 
                                    num_labels=4,  
                                    use_cuda=cuda_available)

    # Student: Distilroberta model
    student_model = Distillation_ClassificationModel(teacher_model = teacher_model,
                                                    model_type="xlmroberta",
                                                    model_name='nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large', # student model is also roberta
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
    df_train = pd.read_csv('data/emotions/EmpatheticPersonas/EN-ZH/emotionlabeled_train.csv')
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

    student_model = run_training(epoch = 20, #epoch,
                                learning_rate = 5e-05, #lr,
                                output_dir = 'distillation/outputs/testing', #f'empathy_classifier/outputs/{str(epoch)}/{str(lr)}', 
                                best_model_dir = 'distillation/best_model', #f'empathy_classifier/best_model/{str(epoch)}/{str(lr)}', 
                                use_early_stopping = True,
                                early_stopping_delta = 0.01,
                                early_stopping_metric_minimize = False,
                                early_stopping_patience = 5,
                                evaluate_during_training_steps = 500, 
                                evaluate_during_training=True,
                                train_df = df_train[['text','labels']],
                                eval_df = df_val[['text','labels']]
                                )

    # evaluate
    print('Student Model Validation Performance')
    evaluate(student_model, df_val)

    print('Student Model Test Performance')
    evaluate(student_model, df_test)

    print('Student Model Native Performance')
    evaluate(student_model, df_native)

    print('Student Model EN Performance')
    evaluate(student_model, df_EN)

    
# LOGS:
# 53581: nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large model, 5e-05, 20 epoch,  batch size = 8
#           loss: KLDiv(log_softmax(student),softmax(teacher)) + student_loss
# 53604: new loss = cross_entropy instead of KLDiv and no * (self.args.temperature ** 2)
# 53614: new loss = include cosineembeddingloss