from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import sys
import numpy as np

#----- Read Data -----
with open('train_p') as f:
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
"""
#----- Standardization of features -----
X_total = np.concatenate((X, X_test), axis=0)
for i in range(X.shape[1]):
	if np.std(X_total[:,i]) != 0:
		X[:,i] = (X[:,i] - np.mean(X_total[:,i]))/np.std(X_total[:,i]);
		X_test[:,i] = (X_test[:,i] - np.mean(X_total[:,i]))/np.std(X_total[:,i]);
"""
"""
#----- Cross Validation Split -----
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in skf.split(X, Y):
	x_train, x_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]

	#----- Define Model -----
	clf = DecisionTreeClassifier(random_state=1, class_weight='balanced')
	#----- Train -----
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print accuracy_score(y_test, y_pred)
"""    
#[10048, 178867, 418211, 34, 791, 8876]             
#[11017, 178827, 418231, 24, 809, 8888]
#[21, 179246, 418081, 24, 1223, 8205] 
#[559, 179258, 418239, 35, 1106, 8141] 
state = 10040
clf = DecisionTreeClassifier(random_state=state)
clf.fit(X, Y)
Y_pred = clf.predict(X_test)
print [np.sum(Y_pred==0), np.sum(Y_pred==1), np.sum(Y_pred==2), np.sum(Y_pred==3), np.sum(Y_pred==4)]
# best: 179122, 418220, 26, 1075, 8336
# approx: <174570, 432209, ...

f = open('pred_DT', 'w')
f.write('id,label\n')
for i in range(len(Y_pred)):
	f.write(str(i+1) + ',' + str(int(Y_pred[i])) + '\n')

