"""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
temp = df[df[df.columns[2]]==df.iloc[0][2]]
temp = temp.iloc[:,3:]
m = temp.as_matrix()
m = m.astype(np.float)
m = m.reshape(1,-1)
"""
import sys
import numpy as np
f = open(sys.argv[1],'r')
temp = f.read().splitlines()
raw=[]
for x in temp:
	raw.append(x.split(','))
f.close()
raw = np.array(raw)
# Finding the first AMB_TEMP to parse the file
find_loc = np.where(raw=='AMB_TEMP')
start_loc = np.array([find_loc[0][0],find_loc[1][0]])
# Replace NR with 0
raw[raw=='NR'] = 0
# Find the set of labels (such as AMB_TEMP)
label = raw[start_loc[0]:(start_loc[0]+18),start_loc[1]]
data = raw[raw[:,start_loc[1]]==label[0]]
data = data[:,start_loc[1]+1:]
data = data.astype(np.float)
data = data.reshape(1,-1)
# further concatenate data with other sets of data corresponding to each label
for i in range(1,label.size,1): # starts from 1 because label[0] is already parsed above
	label = raw[start_loc[0]:(start_loc[0]+18),start_loc[1]]
	temp = raw[raw[:,start_loc[1]]==label[i]]
	temp = temp[:,start_loc[1]+1:]
	temp = temp.astype(np.float)
	temp = temp.reshape(1,-1)
	data = np.vstack((data,temp))
# data[:,:] is now a 2-d array that correspond to each sample
# each row correspond to each label
# each column correspond to the consecutive sample time
# Then we'll have to eat data in chunks, 9 samples as a set of features, then predict the (9+1)th set's PM2.5 


"""
string =  ",".join(map(str,raw))
output = open('ans1.txt','w')
output.write(string)
output.close()
"""