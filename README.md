# multilingual-psychotherapy-chatbot
### Welcome to the SATbot Repository
As part of my MSc Individual Project at Imperial, I have developed a NLP-assisted chatbot web application for Self Attachment Therapy (SAT). The primary focus was to extend previous English-based chatbots to non-English languages. The project involves the use of multilingual transformers (XLM-R) for emotion classification, and transformer text generation models (GPT-2) trained using reinforcement learning with proximal policy optimisation to develop empathetic rewritings that are fluent and semantically accurate during conversation. We also apply Knowledge Distillation, to obtain cheaper, quicker yet performant models that are practical for deployment. 

![satbot-demo](https://user-images.githubusercontent.com/79727686/182863543-fa990870-79a3-4e29-ac94-beaaabc1cbb1.gif)

### Main Folders:
1. <i>data</i><br>
Contains the dataset used to train the machine learning models used by the chatbot.

2. <i>nlp-scripts</i><br>
Contains python scripts used to train NLP classifier models (XLM-Roberta) and generation models (Chinese GPT-2).

3. <i>satbot-app</i><br>
Contains the implementation files of the chatbot web app. 
