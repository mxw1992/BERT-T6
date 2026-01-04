# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:21:01 2025

@author: Lenovo
"""
import pandas as pd

with open("./samples_train_set_0.fasta", 'r') as f:
    content = f.read()
    seq = content.split('>')
del(seq[0])


df = pd.DataFrame(columns=['ID', 'SEQUENCE', 'SEQUENCE_space', 'Label'])
# a = seq[0].split('|')
# print(a)

for i in range(len(seq)):
    a = seq[i].split('|')
    df.loc[i] = [a[0], a[2].strip(), " ".join(a[2].strip()), a[1]]
df.to_csv('./samples_train_set_0.csv')