from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, MaxoutDense
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import sys
import numpy as np
import pickle
# load test data
dir1 = sys.argv[1] + 'test.p'
test = pickle.load(open(dir1,'rb'))
test = np.array(test['data'])
X  = np.zeros((test.shape[0],3,32,32))
for img in range(test.shape[0]):
		X[img] = test[img].reshape(3,32,32)
X = X.astype('float32')
X /= 255
# load json and create model
json_str = sys.argv[2] + ".json"
h5_str = sys.argv[2] + ".h5"
json_file = open(json_str, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(h5_str)
print "Loaded model from disk"

# testing
probs = model.predict(X, verbose=0)
pred = np.argmax(probs, axis=1)

# write to .csv
f = open(sys.argv[3],'w')
f.write("ID,class\n")
for i in range (pred.shape[0]):
	f.write(str(i)+","+str(pred[i])+"\n")
f.close()
print "Prediction written to file: " + sys.argv[3] 
