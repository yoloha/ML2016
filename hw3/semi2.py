from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adam
from sklearn import svm

import sys
import numpy as np
import pickle



print "Loading labeled data..."
dir1 = sys.argv[1] + 'all_label.p'
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
dir2 = sys.argv[1] + 'all_unlabel.p'
all_unlabel = pickle.load(open(dir2,'rb'))
X2  = np.zeros((45000,3,32,32))
for img in range(45000):
	X2[img] = np.array(all_unlabel[img]).reshape(3,32,32)
X2 = X2.astype('float32')
X2 /= 255

print "Loading test data..."
dir2 = sys.argv[1] + 'test.p'
test = pickle.load(open(dir2,'rb'))
X3  = np.zeros((10000,3,32,32))
for img in range(10000):
	X3[img] = np.array(all_unlabel[img]).reshape(3,32,32)
X3 = X3.astype('float32')
X3 /= 255
X2 = np.concatenate((X2,X3), axis=0)



X3 = np.concatenate((X,X2), axis=0)

input_img = Input(shape=(3, 32, 32))

x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(input_img)
x = MaxPooling2D((2, 2), border_mode='same', dim_ordering="th")(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(x)
x = MaxPooling2D((2, 2), border_mode='same', dim_ordering="th")(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(x)
encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering="th")(x)

# at this point the representation is (16, 4, 4) i.e. 256-dimensional

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(encoded)
x = UpSampling2D((2, 2), dim_ordering="th")(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(x)
x = UpSampling2D((2, 2), dim_ordering="th")(x)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering="th")(x)
x = UpSampling2D((2, 2), dim_ordering="th")(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same', dim_ordering="th")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X3, X3, nb_epoch=5, batch_size=50, verbose=1)

encoder = Model(input=input_img, output=encoded)

X_code = encoder.predict(X)
X_code = X_code.reshape(X.shape[0],-1)
X2_code = encoder.predict(X2)
X2_code = X2_code.reshape(X2.shape[0],-1)
print X_code.shape
print X2_code.shape

clf = svm.SVC()
clf.fit(X_code, Y)
Y2 = np.array(clf.predict(X2_code))
Y2 = Y2.reshape(-1,1) 
X3 = np.concatenate((X,X2), axis=0)
Y3 = np.concatenate((Y,Y2), axis=0)

# Only using the first 10000 data, i.e. 5000 labeled and 5000 unlabeled
X3 = X3[0:10000]
Y3 = Y3[0:10000]

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
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])# sparse

model.fit(X3, Y3, batch_size=50, nb_epoch=500, verbose=1)

# serialize model to JSON
model_json = model.to_json()
json_str = sys.argv[2] + ".json"
h5_str = sys.argv[2] + ".h5"
with open(json_str, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(h5_str)
print("Saved model to disk")