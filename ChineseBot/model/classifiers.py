import torch
from simpletransformers.classification import ClassificationModel

# emotion classification labels corr to the language
labels = {
  "中文(ZH)": {0:"悲伤", 1:"愤怒", 2:"快乐", 3:"焦虑"},
  "English(EN)": {0:"Sad", 1:"Angry", 2:"Happy/Content", 3:"Anxious/Scared"}
}

# load emotion classifier
with torch.no_grad():
  cuda_available = torch.cuda.is_available()
  # Empathy Classifier
  EMOTION_CLASSIFIER_NAME = 'emotion-classifier-distilled'
  emotion_classifier = ClassificationModel(model_type="xlmroberta", 
                                          model_name=EMOTION_CLASSIFIER_NAME, 
                                          num_labels=4,
                                          use_cuda=cuda_available)

def get_emotion(user_input, language):
  '''
  Classifies and returns the underlying emotion of a text string
  '''
  y_pred, _ = emotion_classifier.predict([user_input])
  labels_map_in_lang = labels[language]

  return labels_map_in_lang[y_pred[0]]



