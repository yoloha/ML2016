from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
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

"""
#----- Cross Validation Split -----
skf = StratifiedKFold(n_splits=5, shuffle=True)
for train_index, test_index in skf.split(X, Y):
	X_train, X_test = X[train_index], X[test_index]
	Y_train, Y_test = Y[train_index], Y[test_index]

	#----- Define Model -----
	clf = MLPClassifier(solver='adam', alpha=1e-5, early_stopping=True, verbose=True,
					hidden_layer_sizes=(50, 20, 10, 5, 2), random_state=1)
	#----- Train -----
	clf.fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)
	print accuracy_score(Y_test, Y_pred)
"""


clf = MLPClassifier(solver='adam', alpha=1e-5, early_stopping=True, verbose=True,
					hidden_layer_sizes=(50, 20, 10, 5, 2), random_state=1)
clf.fit(X, Y)



Y_pred = clf.predict(X_test)

f = open('pred', 'w')
f.write('id,label\n')
for i in range(len(Y_pred)):
	f.write(str(i+1) + ',' + str(int(Y_pred[i])) + '\n')
