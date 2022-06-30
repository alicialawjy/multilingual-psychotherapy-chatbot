import os
import pandas as pd
from prism import Prism

prism = Prism(model_dir=os.environ['MODEL_DIR'], lang='zh') # lang is target lang
print('Prism identifier:', prism.identifier())

df_zh= pd.read_csv('EP_empathy_2144_ZH.csv')
df_en= pd.read_csv('EP_empathy_2144_EN.csv')

src = list(df_en['rewriting']) # source language
cand = list(df_zh['rewriting']) # output in target language

score = prism.score(cand=cand, src=src, segment_scores=True)
df_prism_score = pd.DataFrame(list(zip(cand,score)), columns=['rewriting','score'])
df_prism_score.to_csv('prism-master/score.csv')
print('System-level QE-as-metric:', prism.score(cand=cand, src=src))
print('Segment-level QE-as-metric:', prism.score(cand=cand, src=src, segment_scores=True))