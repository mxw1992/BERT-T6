import pandas as pd
import os
from sklearn.utils import shuffle
 
data = pd.read_csv('./samples_train_set_0.csv')
data = shuffle(data) # 打乱
 
data.to_csv('./samples_train_set_0_shuffle.csv')