### Emotion Classifier Datasets
```
emotion-classifier
├─ ECM
|  ├─ ECM_test.csv
|  ├─ ECM_train.csv
|  ├─ ECM_formatting.ipynb
|  └─ ECM.csv
└─ EmpatheticPersonas
    ├─ EN
    |   ├─ emotionlabeled_full.csv
    |   ├─ emotionlabeled_test.csv
    |   ├─ emotionlabeled_train.csv
    |   └─ emotionlabeled_val.csv
    ├─ ZH
    |   ├─ emotionlabeled_full.csv
    |   ├─ emotionlabeled_test.csv
    |   ├─ emotionlabeled_train.csv
    |   └─ emotionlabeled_val.csv
    ├─ EP_codeswitch.csv
    ├─ EP_native.csv
    ├─ EP_train_augmented.csv
    └─ EP_train.csv
```

- <i>ECM</i><br>
This is the Emotional Chatting Machine (ECM) Dataset [1], which is a native Mandarin emotional dataset. Folder contains the data, including their respective train-test splits and the ipynb file used to do the formatting.

- <i>EmpatheticPersonas</i><br>
    1. EN and ZH Directory<br>
    Contains both EN and ZH versions of the dataset respectively, including their respective train-test-val splits.

    2. EP_codeswitch.csv <br>
    100 instances of EN-ZH codeswitched sentences (for model evaluation only)

    3. EP_native.csv <br>
    120 instances of native ZH sentences (for model evaluation only)

    4. EP_train.csv <br>
    The concatenation of emotionlabeled_train.csv datasets in the EN and ZH folder.

    5. EP_train_augmented.csv <br>
    The final training file used to train the emotion classifier. Extended the training set by combining sentences of the same emotion label together to generate more sentence variation.

#### Reference:
[1] Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu, and Bing Liu. Emotional chatting machine: Emotional conversation generation with internal and external memory. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), Apr. 2018. doi: 10.1609/aaai.v32i1.11325. URL https://ojs.aaai.org/index.php/AAAI/article/view/11325. 