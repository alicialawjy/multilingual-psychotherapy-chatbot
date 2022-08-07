## Empathy Classifier Datasets
```
empathy-classifier
├─ EN_labelled
|  ├─ EP_EN_full.csv
|  ├─ EP_EN_test.csv
|  └─ EP_EN_train.csv
├─ ZH_labelled
|   ├─ EP_ZH_full.csv
|   ├─ EP_ZH_test.csv
|   ├─ EP_ZH_train.csv
|   └─ EP_ZH_val.csv
├─ empathy.ipynb
└─ EP_train.csv
```

All dataset in this folder are from the <i>EmpatheticPersonas</i> dataset.

1. EN_labelled and ZH_labelled Directory<br>
Contains empathetic rewritings that have been labelled for empathy in both EN and ZH versions of the dataset respectively, including their respective train-test-val splits.

2. EP_train.csv <br>
The final dataset used to train the empathy classifier. Comprises of EP_EN_train.csv + EP_ZH_train.csv .

3. empathy.ipynb <br>
The jupyter notebook used to clean, format and split the datasets for empathy classification.

#### Reference:
[1] Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu, and Bing Liu. Emotional chatting machine: Emotional conversation generation with internal and external memory. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), Apr. 2018. doi: 10.1609/aaai.v32i1.11325. URL https://ojs.aaai.org/index.php/AAAI/article/view/11325. 