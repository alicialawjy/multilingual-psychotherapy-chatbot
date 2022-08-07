## Classification Folder: This folder contains the dataset used to train the various classifiers used in the project.

The SATbot involves the use of 3 classifiers:
1. emotion-classifier <br>
This classifier is <b>used directly</b> by the chatbot to predict user's emotions during conversation.

2. empathy-classifier <br>
Used during GPT-2 training of the empathetic rewriting task. Classifier assesses whether a sentence is empathetic or not.

3. semantic-classifier <br>
Used during GPT-2 training (transformer reinforcement learning only) of the empathetic rewriting task. Classifier assesses whether a generated sentence matches the semantics of the base utterance it is meant to rewrite.