### Data Folder: This folder contains all the data files used to train the transformer machine learning models used by the chatbot.

They have been split into 2 folders:
1. classification <br>
Data used to train classification models (XLM-Roberta), and 
2. generation <br> 
Data used to train the generative model (Chinese GPT-2).

These datasets have been cleaned and formatted from the <i>EmpatheticPersonas</i> dataset:
- <i>empatheticPersonasEN.csv</i>: the original dataset sourced from Alazraki [1] in English.
- <i>empatheticPersonasZH.csv</i>: translated version of the EN dataset by Hu [2] into Mandarin.

Formatting done using <i>EP_formatting.ipynb</i>
<br><br>
#### Reference:
[1] Lisa Alazraki (2021) A deep-learning assisted empathetic guide for self-attachment
therapy. Master’s thesis, Imperial College London. <br>
[2] Ruoyu Hu (2022) A multi-language virtual psychotherapy chatbot. Master’s thesis,
Imperial College London