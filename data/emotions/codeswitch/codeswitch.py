#-------------------------------------------------------------------------------
# Name:        Evaluation for NLPCC-2018 ShareTask, Code-switching prediction
# Purpose:     evaluate the results of code-switching prediction
#
# Author:      zhongqing
#
# Created:     13/02/2018
# Copyright:   (c) zhongqing 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from __future__ import division
from sys import argv
import pandas as pd

class Post:
    def __init__(self,content,emotion):
        self.text = content
        self.label = emotion

def read_posts(fpath):
    posts=[]
    postBegin=False
    post=[]
    d={}
    #print fpath
    for line in open(fpath,'rb'):
        line=line.strip().decode('utf-8')
        if len(line)>0:
            if line[:10]=='<Tweet id=':
                postBegin=True
                continue
            if line=='</Tweet>':
                postBegin=False
                content=post[5*3+1]
                emotion=[post[1*3+1], post[2*3+1],post[1],post[3*3+1]] #[sad,anger,happy,fear]
                
                if content not in d and 'T' in emotion:
                    posts.append(Post(content,emotion.index('T')))
                    d[content]=0

                post=[]
                continue
            if postBegin:
                post.append(line)
    #print 'length of posts',len(posts)
    return posts

if __name__ == '__main__':
    posts_dev = read_posts('data/dev.txt')
    posts_train = read_posts('data/train.txt')
    label = []
    text = []
    for post in posts_dev:
        text.append(post.text)
        label.append(post.label)

    for post in posts_train:
        text.append(post.text)
        label.append(post.label)

    print(len(label))

    # create the organised df
    df_codeswitch = pd.DataFrame()
    df_codeswitch['text']=text
    df_codeswitch['labels']=label

    df_codeswitch.to_csv('data/codeswitch.csv')