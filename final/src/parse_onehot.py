import sys
import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder
#----- Read Data -----
with open('train_p_nodup') as f:
    raw = f.read().splitlines()
data_tr = []
for line in raw:
    data_tr.append(line.split(','))
data_tr = np.array(data_tr)
data_tr = data_tr.astype(float)

with open('test_p') as f:
    raw = f.read().splitlines()
data_te = []
for line in raw:
    data_te.append(line.split(','))
data_te = np.array(data_te)
data_te = data_te.astype(float)
print data_tr.shape

enc = OneHotEncoder(sparse=False)
enc.fit(data_tr[:,1:4])
enc.fit(data_te[:,1:4])
data_tr_app = enc.transform(data_tr[:,1:4])
data_te_app = enc.transform(data_te[:,1:4])

data_tr1 = data_tr[:,0].reshape((-1,1))
data_tr2 = data_tr[:,4:]
data_te1 = data_te[:,0].reshape((-1,1))
data_te2 = data_te[:,4:]

data_tr = np.concatenate((data_tr1,data_tr_app), axis=1)
data_tr = np.concatenate((data_tr,data_tr2), axis=1)
data_te = np.concatenate((data_te1,data_te_app), axis=1)
data_te = np.concatenate((data_te,data_te2), axis=1)

print data_tr.shape

# Output parsed files
data_tr = data_tr.astype(str)
data_te = data_te.astype(str)
f = open('train2_p_nodup', 'w')
for i in range(data_tr.shape[0]):
	f.write(','.join(data_tr[i,:]) + '\n')

f = open('test2_p', 'w')
for i in range(data_te.shape[0]):
	f.write(','.join(data_te[i,:]) + '\n')




