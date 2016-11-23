import sys
import numpy as np
def sigmoid(z):
    return 1.0/(1 + np.power(np.e,-z))
def predict(model, X):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
    # Forward propagation
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    a3 = sigmoid(z3)
    z4 = a3.dot(W4) + b4
    exp_scores = np.exp(z4)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)
#-------------------Acquire model-------------------
f = open(sys.argv[1],'r')
temp = f.read().splitlines()
raw=[]
for x in temp:
    raw.append(x.split(','))
f.close()
raw = np.array(raw)
raw[raw=="0."] = 0

temp = np.array(raw[0])
hid_sizes = temp.astype(np.int)
raw = raw[1:]

in_dim = 57
out_dim = 2
W1 = np.random.randn(in_dim, hid_sizes[0]) / np.sqrt(in_dim)
b1 = np.zeros((1, hid_sizes[0]))
W2 = np.random.randn(hid_sizes[0], hid_sizes[1]) / np.sqrt(hid_sizes[0])
b2 = np.zeros((1, hid_sizes[1]))
W3 = np.random.randn(hid_sizes[1], hid_sizes[2]) / np.sqrt(hid_sizes[1])
b3 = np.zeros((1, hid_sizes[2]))
W4 = np.random.randn(hid_sizes[2], out_dim) / np.sqrt(hid_sizes[2])
b4 = np.zeros((1, out_dim))


for i in range (W1.shape[0]):
	temp = np.array(raw[0])
	W1[i] = temp.astype(np.float)
	raw = raw[1:]
for i in range (b1.shape[0]):
	temp = np.array(raw[0])
	b1[i] = temp.astype(np.float)
	raw = raw[1:]
for i in range (W2.shape[0]):
	temp = np.array(raw[0])
	W2[i] = temp.astype(np.float)
	raw = raw[1:]
for i in range (b2.shape[0]):
	temp = np.array(raw[0])
	b2[i] = temp.astype(np.float)
	raw = raw[1:]
for i in range (W3.shape[0]):
	temp = np.array(raw[0])
	W3[i] = temp.astype(np.float)
	raw = raw[1:]
for i in range (b3.shape[0]):
	temp = np.array(raw[0])
	b3[i] = temp.astype(np.float)
	raw = raw[1:]
for i in range (W4.shape[0]):
	temp = np.array(raw[0])
	W4[i] = temp.astype(np.float)
	raw = raw[1:]
for i in range (b4.shape[0]):
	temp = np.array(raw[0])
	b4[i] = temp.astype(np.float)
	raw = raw[1:]


model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4}
#-------------------Acquire testing data-------------------
f = open(sys.argv[2],'r')
temp = f.read().splitlines()
raw=[]
for x in temp:
    raw.append(x.split(','))
f.close()
raw = np.array(raw)
M = raw[:,1:]# 1st col to the end
M = M.astype(np.float)
# Standardization of M
for i in range(57):
	M[:,i] = (M[:,i] - np.mean(M[:,i]))/np.std(M[:,i])
est = predict(model, M)
#-------------------Output-------------------
f = open(sys.argv[3],'w')
f.write("id,label\n")
for i in range (est.shape[0]):
	f.write(str(i+1)+","+str(np.sum(est[i]))+"\n")
f.close()
