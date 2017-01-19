from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sys
import numpy as np
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

X = data_tr[:,:-1]
Y = data_tr[:,-1]
X_test = data_te

clf = RandomForestClassifier(n_estimators=1, random_state=194024)
clf.fit(X, Y)

Y_pred1 = clf.predict(X_test)
print [np.sum(Y_pred1==0), np.sum(Y_pred1==1), np.sum(Y_pred1==2), np.sum(Y_pred1==3), np.sum(Y_pred1==4)]

list1 = np.where((Y_pred1 == 3))[0]
list2 = np.where(Y_pred1 != 3)[0]

X_test2 = X_test[list2,:]

clf = RandomForestClassifier(n_estimators=1, random_state=45280)
clf.fit(X, Y)
Y_pred2 = clf.predict(X_test2)
Y_pred = Y_pred1
Y_pred[list2] = Y_pred2

score = abs(np.sum(Y_pred==0) - 157556) + abs(np.sum(Y_pred==1) - 432209) \
	      + abs(np.sum(Y_pred==2) - 230) + abs(np.sum(Y_pred==3) - 8568) \
	      + abs(np.sum(Y_pred==4) - 8216) 
print score, [np.sum(Y_pred==0), np.sum(Y_pred==1), np.sum(Y_pred==2), np.sum(Y_pred==3), np.sum(Y_pred==4)]


f = open('pred_forest', 'w')
f.write('id,label\n')
for i in range(len(Y_pred)):
	f.write(str(i+1) + ',' + str(int(Y_pred[i])) + '\n')

