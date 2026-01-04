# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 13:27:48 2025

@author: Lenovo
"""
from sklearn.model_selection import train_test_split

f = open("./neg_samples.fasta", 'r')
content = f.read()
f.close()

seq = content.split('>')
X_train, X_test = train_test_split(seq, test_size=0.2, random_state=0)
print(len(X_test))

for i in range(0,len(X_test)-1):
    if(X_test[i] ==''):
        del X_test[i]
        
for i in range(0,len(X_test)):
    m = X_test[i].split('\n')
    X_test[i] = '>' + m[0] + '\n' + m[1] + '\n'

content1 = ''
for j in range(0,len(X_test)):
    content1 = content1 + X_test[j]
f = open("./neg_samples_test_set_0.fasta",'w')
f.write(content1)
f.close()

for i in range(0,len(X_train)-1):
    if(X_train[i] ==''):
        del X_train[i]
print(len(X_train))


for i in range(0,len(X_train)):
    n = X_train[i].split('\n')
    X_train[i] = '>' + n[0] + '\n' + n[1] + '\n'
content2 = ''
for j in range(0,len(X_train)):
    content2 = content2 + X_train[j]

f = open("./neg_samples_train_set_0.fasta",'w')
f.write(content2)
f.close()

