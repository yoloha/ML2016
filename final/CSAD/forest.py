from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
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

#----- Standardization of features -----
for i in range(X.shape[1]):
	if np.std(X[:,i]) != 0:
		X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i]);
for i in range(X_test.shape[1]):
	if np.std(X_test[:,i]) != 0:
		X_test[:,i] = (X_test[:,i] - np.mean(X_test[:,i]))/np.std(X_test[:,i]);


#----- Cross Validation Split -----
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in skf.split(X, Y):
	x_train, x_test = X[train_index], X[test_index]
	y_train, y_test = Y[train_index], Y[test_index]

	#----- Define Model -----
	clf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=8)
	#----- Train -----
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print accuracy_score(y_test, y_pred)



clf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=8)
clf.fit(X, Y)

Y_pred = clf.predict(X_test)

f = open('pred', 'w')
f.write('id,label\n')
for i in range(len(Y_pred)):
	f.write(str(i+1) + ',' + str(int(Y_pred[i])) + '\n')

