from sklearn.tree import DecisionTreeClassifier

import sys
import numpy as np

#----- Read Data -----
with open('../data/train_p_nodup') as f:
    raw = f.read().splitlines()
data_tr = []
for line in raw:
    data_tr.append(line.split(','))
data_tr = np.array(data_tr)
data_tr = data_tr.astype(float)

with open('../data/test_p') as f:
    raw = f.read().splitlines()
data_te = []
for line in raw:
    data_te.append(line.split(','))
data_te = np.array(data_te)
data_te = data_te.astype(float)


X = data_tr[:,:-1]
Y = data_tr[:,-1]
X_test = data_te

state = 10040
clf = DecisionTreeClassifier(random_state=state)
clf.fit(X, Y)
Y_pred = clf.predict(X_test)
print [np.sum(Y_pred==0), np.sum(Y_pred==1), np.sum(Y_pred==2), np.sum(Y_pred==3), np.sum(Y_pred==4)]

f = open('../pred_DT', 'w')
f.write('id,label\n')
for i in range(len(Y_pred)):
	f.write(str(i+1) + ',' + str(int(Y_pred[i])) + '\n')

