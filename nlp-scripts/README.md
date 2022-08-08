### NLP Scripts
This folder contains all the training, inference and evaluation scripts used to train the various nlp models.

1. classify.py <br>
Script used to get evaluation/ predictions for classifiers.

2. run_classifier.py <br>
Training script used to train the (i) emotion, (ii) empathy and (iii) semantic classifiers.

3. run_distillation.py <br>
Training script used to run knowledge distillation (using Triple Loss) for the emotion classifier.

4. run_generation.py <br>
Training script used to train a text generation model. Includes: <br>
(i) supervised training (for warm startup), and <br> 
(ii) reinforcment training using Proximal Policy Optimisation (PPO).
    PPO implementation follows the paper **"Fine-Tuning Language Models from Human Preferences"** by D. Ziegler et al. \[[paper](https://arxiv.org/pdf/1909.08593.pdf), [code](https://github.com/openai/lm-human-preferences)].

5. run_inference.py <br>
Obtains the output from the text generation model.