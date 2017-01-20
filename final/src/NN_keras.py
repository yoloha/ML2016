from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping

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
Y = np.expand_dims(Y,-1)
print X.shape
print Y.shape
X_test = data_te
#----- Standardization of features -----
X_total = np.concatenate((X, X_test), axis=0)
for i in range(X.shape[1]):
	if np.std(X_total[:,i]) != 0:
		X[:,i] = (X[:,i] - np.mean(X_total[:,i]))/np.std(X_total[:,i]);
		X_test[:,i] = (X_test[:,i] - np.mean(X_total[:,i]))/np.std(X_total[:,i]);

#----- Define Model -----
def build_model(): 
	model = Sequential()
	model.add(Dense(512, input_dim=X.shape[1]))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(128))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(16))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(5))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

#----- Using the whole set of data to do final prediction -----
model = build_model()
model.fit(X, Y, batch_size=1000, nb_epoch=5, verbose=1)
probs = model.predict(X_test, verbose=0)
Y_pred = np.argmax(probs, axis=1)
print [np.sum(Y_pred==0), np.sum(Y_pred==1), np.sum(Y_pred==2), np.sum(Y_pred==3), np.sum(Y_pred==4)]


f = open('../data/pred_NN', 'w')
f.write('id,label\n')
for i in range(len(Y_pred)):
	f.write(str(i+1) + ',' + str(int(Y_pred[i])) + '\n')
