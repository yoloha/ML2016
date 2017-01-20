from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

print X.shape
print Y.shape
X_test = data_te
#----- Standardization of features -----
X_total = np.concatenate((X, X_test), axis=0)
for i in range(X.shape[1]):
	if np.std(X_total[:,i]) != 0:
		X[:,i] = (X[:,i] - np.mean(X_total[:,i]))/np.std(X_total[:,i]);
		X_test[:,i] = (X_test[:,i] - np.mean(X_total[:,i]))/np.std(X_total[:,i]);


clf = MLPClassifier(solver='adam', early_stopping=True, verbose=True,
					hidden_layer_sizes=(5, 3), random_state=1)
clf.fit(X, Y)

Y_pred = clf.predict(X_test)
print [np.sum(Y_pred==0), np.sum(Y_pred==1), np.sum(Y_pred==2), np.sum(Y_pred==3), np.sum(Y_pred==4)]
# best: 179122, 418220, 26, 1075, 8336
# approx: <174570, 432209, ...

f = open('../pred_NN', 'w')
f.write('id,label\n')
for i in range(len(Y_pred)):
	f.write(str(i+1) + ',' + str(int(Y_pred[i])) + '\n')

