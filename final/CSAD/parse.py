import sys
import numpy as np
import pandas

# Read training data
with open('train') as f:
    raw = f.read().splitlines()
data_tr = []
for line in raw:
    data_tr.append(line.split(','))
data_tr = np.array(data_tr)
for i in range(data_tr.shape[0]):
    data_tr[i,-1] = data_tr[i,-1].replace('.','')
with open('training_attack_types.txt') as f:
    raw = f.read().splitlines()
type_map = [['normal', 0]]
for line in raw:
    type_map.append(line.split())

# Read Testing data
with open('test.in') as f:
    raw = f.read().splitlines()
data_te = []
for line in raw:
    data_te.append(line.split(','))
data_te = np.array(data_te)


# Mapping the last column of training data from names to 0, 1, 2, 3, 4
type_map = np.array(type_map)
attk_map =  np.array(['normal', 'dos', 'u2r', 'r2l', 'probe'])
for i in range(1,type_map.shape[0]):
    type_map[i,1] = np.where(attk_map == type_map[i,1])[0][0]
print type_map
for i in range(type_map.shape[0]):
    data_tr[data_tr==type_map[i,0]] = type_map[i,1]

# Mapping some columns of both training and testing data from strings to numbers
mismatch = 0
str_list = np.array([1,2,3])
for i in str_list:
	uniq_list =  np.unique(data_tr[:,i])
	print uniq_list
	for j in range(data_tr.shape[0]):
		data_tr[j,i] = np.where(uniq_list == data_tr[j,i])[0][0]
	for j in range(data_te.shape[0]):
		if data_te[j,i] in uniq_list:
			data_te[j,i] = np.where(uniq_list == data_te[j,i])[0][0]
		else:
			data_te[j,i] = len(uniq_list)
			mismatch += 1
print mismatch



df = pandas.DataFrame(data_tr, index=range(data_tr.shape[0]), columns=range(data_tr.shape[1]))
df2 = df.drop_duplicates()
data_tr = df2.as_matrix()




# Output parsed files
data_tr = data_tr.astype(str)
data_te = data_te.astype(str)
f = open('train_p', 'w')
for i in range(data_tr.shape[0]):
	f.write(','.join(data_tr[i,:]) + '\n')

f = open('test_p', 'w')
for i in range(data_te.shape[0]):
	f.write(','.join(data_te[i,:]) + '\n')




