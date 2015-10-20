import numpy as np
import pandas as pd

# Load data
print("Load the test data using pandas")
test  = pd.read_csv("../input/test.csv")
#print test.shape
#print test.iloc[0]
#print test.iloc[test.shape[0]-1]

fix_interval = int(test.shape[0]/5.)
all_index = range(0,test.shape[0])

for i in range(0,5):
    print i
    print i*fix_interval
    print (i+1)*fix_interval
    if i==4:
        temp_index = range(i*fix_interval,test.shape[0])
    else:
        temp_index = range(i*fix_interval,(i+1)*fix_interval)

    temp_data_1 = test.iloc[temp_index]
    temp_data_2 = test.iloc[list(set(all_index)-set(temp_index))]

    temp_data_1.to_csv("./input/testing_eval_%i.csv" %i,index=False )
    temp_data_2.to_csv("./input/testing_%i.csv" %i,index=False )
