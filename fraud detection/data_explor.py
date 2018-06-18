# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:11:03 2018

@author: fzhan
"""
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('train_sample.csv')
data3=pd.DataFrame(data['ip'].value_counts())
data4=data[['ip','is_attributed']].groupby('ip').sum()
data4['is_attributed'].value_counts()




plt.hist(data4['is_attributed'])
plt.show()