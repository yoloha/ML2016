from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
import sys
import numpy as np
import pickle



print "Loading labeled data..."
dir1 = sys.argv[1] + '/all_label.p'
all_label = pickle.load(open(dir1,'rb'))
X  = np.zeros((5000,3,32,32))
Y  = np.zeros((5000,1))
for cl in range(10):
	for img in range(500):
		X[cl*500+img] = np.array(all_label[cl][img]).reshape(3,32,32)
		Y[cl*500+img] = cl
X = X.astype('float32')
X /= 255

print "Loading unlabeled data..."
dir2 = sys.argv[1] + '/all_unlabel.p'
all_unlabel = pickle.load(open(dir2,'rb'))
X2  = np.zeros((45000,3,32,32))
Y2  = np.zeros((45000,1))
for img in range(45000):
	X2[img] = np.array(all_unlabel[img]).reshape(3,32,32)
X2 = X2.astype('float32')
X2 /= 255


print "Loading test data..."
dir2 = sys.argv[1] + '/test.p'
test = pickle.load(open(dir2,'rb'))
X3  = np.zeros((10000,3,32,32))
Y3  = np.zeros((10000,1))
for img in range(10000):
	X3[img] = np.array(all_unlabel[img]).reshape(3,32,32)
X3 = X3.astype('float32')
X3 /= 255
X2 = np.concatenate((X2,X3), axis=0)

num_f = 64
model = Sequential()
model.add(Convolution2D(num_f,3,3, input_shape=X.shape[1:], border_mode="same", dim_ordering="th"))
model.add(Activation('relu'))
model.add(Convolution2D(num_f,3,3,  border_mode="same", dim_ordering="th"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Dropout(0.5))

model.add(Convolution2D(num_f*2,3,3,  border_mode="same", dim_ordering="th"))
model.add(Activation('relu'))
model.add(Convolution2D(num_f*2,3,3,  border_mode="same", dim_ordering="th"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Dropout(0.5))

model.add(Convolution2D(num_f*4,3,3,  border_mode="same", dim_ordering="th"))
model.add(Activation('relu'))
model.add(Convolution2D(num_f*4,3,3,  border_mode="same", dim_ordering="th"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])# sparse

X_train = X
Y_train = Y

model.fit(X_train, Y_train, batch_size=50, nb_epoch=500, verbose=1)

for i in range(8):
	probs = np.array(model.predict(X2, verbose=1))
	max_prob = np.amax(probs, axis=1)
	Y2 = np.argmax(probs, axis=1)
	index = sorted(range(len(max_prob)), key=lambda i: max_prob[i])[-5000:]
	X2_sub = np.array([X2[i] for i in index])
	Y2_sub = np.array([Y2[i] for i in index])
	Y2_sub = Y2_sub.reshape(-1,1)
	X_train = np.concatenate((X_train,X2_sub), axis=0)
	Y_train = np.concatenate((Y_train,Y2_sub), axis=0)
	X2 = np.array([i for j, i in enumerate(X2) if j not in index])

	num_f = 64
	model = Sequential()
	model.add(Convolution2D(num_f,3,3, input_shape=X.shape[1:], border_mode="same", dim_ordering="th"))
	model.add(Activation('relu'))
	model.add(Convolution2D(num_f,3,3,  border_mode="same", dim_ordering="th"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
	model.add(Dropout(0.5))

	model.add(Convolution2D(num_f*2,3,3,  border_mode="same", dim_ordering="th"))
	model.add(Activation('relu'))
	model.add(Convolution2D(num_f*2,3,3,  border_mode="same", dim_ordering="th"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
	model.add(Dropout(0.5))

	model.add(Convolution2D(num_f*4,3,3,  border_mode="same", dim_ordering="th"))
	model.add(Activation('relu'))
	model.add(Convolution2D(num_f*4,3,3,  border_mode="same", dim_ordering="th"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(1024))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1024))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])# sparse

	model.fit(X_train, Y_train, batch_size=50, nb_epoch=100, verbose=1)

# serialize model to JSON
model_json = model.to_json()
json_str = sys.argv[2] + ".json"
h5_str = sys.argv[2] + ".h5"
with open(json_str, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(h5_str)
print("Saved model to disk")