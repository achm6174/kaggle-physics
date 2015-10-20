"""
@author: achm

Splite data for training and evaluation set with 5 folds
"""

import numpy as np
import pandas as pd

# Load data
print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")

train = train.sort(['min_ANNmuon'],ascending=[True])
end = sum(train['min_ANNmuon'] > 0.4)
start = sum(train['min_ANNmuon'] <= 0.4)

for i in range(5):
    temp_index = range(start+i+1,start+end-1,5)
    temp_data_1 = train.iloc[temp_index]
    temp_data_2 = train.iloc[list(set(range(0,start+end-1))-set(temp_index))]

    temp_data_1.to_csv("./input/training_eval_%i.csv" %i,index=False )
    temp_data_2.to_csv("./input/training_%i.csv" %i,index=False )
