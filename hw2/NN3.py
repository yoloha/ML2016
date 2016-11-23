import sys
import numpy as np
def sigmoid(z):
    return 1.0/(1 + np.power(np.e,-z))
def calculate_loss(X, Y, model):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
    num_examples = X.shape[0] # training set size
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    z3 = a2.dot(W3) + b3
    a3 = sigmoid(z3)
    z4 = a3.dot(W4) + b4
    exp_scores = np.exp(z4)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    correct_logprobs = -np.log(probs[range(num_examples), Y])
    data_loss = np.sum(correct_logprobs)
    return 1.0/num_examples * data_loss
# Helper function to predict an output (0 or 1)
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
# This function learns parameters for the neural network and returns the model.
# - hid_sizes: Number of nodes in the hidden layer
# - it_max: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X, Y, hid_sizes,rate=0.1, lam=1, it_max=20000):
    # Static Parameters
    num_examples = X.shape[0]
    in_dim = X.shape[1]
    out_dim = 2 
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(in_dim, hid_sizes[0]) / np.sqrt(in_dim)
    b1 = np.zeros((1, hid_sizes[0]))
    W2 = np.random.randn(hid_sizes[0], hid_sizes[1]) / np.sqrt(hid_sizes[0])
    b2 = np.zeros((1, hid_sizes[1]))
    W3 = np.random.randn(hid_sizes[1], hid_sizes[2]) / np.sqrt(hid_sizes[1])
    b3 = np.zeros((1, hid_sizes[2]))
    W4 = np.random.randn(hid_sizes[2], out_dim) / np.sqrt(hid_sizes[2])
    b4 = np.zeros((1, out_dim))
    model = {}
    ada_W1 = np.zeros((in_dim, hid_sizes[0]))
    ada_b1 = np.zeros((1, hid_sizes[0]))
    ada_W2 = np.zeros((hid_sizes[0], hid_sizes[1]))
    ada_b2 = np.zeros((1, hid_sizes[1]))
    ada_W3 = np.zeros((hid_sizes[1], hid_sizes[2]))
    ada_b3 = np.zeros((1, hid_sizes[2]))
    ada_W4 = np.zeros((hid_sizes[2], out_dim))
    ada_b4 = np.zeros((1, out_dim))
    eps = 1e-6
    for it in range(it_max):
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
        # Backpropagation
        delta4 = probs
        delta4[range(num_examples), Y] -= 1
        dW4 = (a3.T).dot(delta4)
        db4 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = delta4.dot(W4.T) * a3 * (1 - a3)
        dW3 = (a2.T).dot(delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W3.T) * a2 * (1 - a2)
        dW2 = (a1.T).dot(delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = delta2.dot(W2.T) * a1 * (1 - a1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0)
        # Regularization
        dW4 += lam * W4
        dW3 += lam * W3
        dW2 += lam * W2
        dW1 += lam * W1
        # Adagrad
        ada_W1 = np.sqrt(np.square(ada_W1) + np.square(dW1))
        ada_b1 = np.sqrt(np.square(ada_b1) + np.square(db1))
        ada_W2 = np.sqrt(np.square(ada_W2) + np.square(dW2))
        ada_b2 = np.sqrt(np.square(ada_b2) + np.square(db2))
        ada_W3 = np.sqrt(np.square(ada_W3) + np.square(dW3))
        ada_b3 = np.sqrt(np.square(ada_b3) + np.square(db3))
        ada_W4 = np.sqrt(np.square(ada_W4) + np.square(dW4))
        ada_b4 = np.sqrt(np.square(ada_b4) + np.square(db4))
        # Gradient descent parameter update
        W1 += -rate * dW1/(ada_W1 + eps)
        b1 += -rate * db1/(ada_b1 + eps)
        W2 += -rate * dW2/(ada_W2 + eps)
        b2 += -rate * db2/(ada_b2 + eps)
        W3 += -rate * dW3/(ada_W3 + eps)
        b3 += -rate * db3/(ada_b3 + eps)
        W4 += -rate * dW4/(ada_W4 + eps)
        b4 += -rate * db4/(ada_b4 + eps)
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4}
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if it % 1000 == 0 or it == it_max -1:
            print "[ %5d"%it,"/",it_max,"]      Loss = %.8f"%calculate_loss(X,Y,model)
    return model
#-------------------Preprocessing Data-------------------
f = open(sys.argv[1],'r')
temp = f.read().splitlines()
raw=[]
for x in temp:
    raw.append(x.split(','))
f.close()
raw = np.array(raw)
M = raw[:,1:-1]# 1st col to the 2nd col from the back
Y = raw[:,-1]# last col is the answer
M = M.astype(np.float)
Y = Y.astype(np.int)
# Standardization of M
for i in range(57):
	M[:,i] = (M[:,i] - np.mean(M[:,i]))/np.std(M[:,i]);
# Best (11,5,3), 0.1, 4, 0.05991
hid = (11,5,3)
rate = 0.1
lam = 4
it_max = 20000

M1 = M[0:3800,:]
M2 = M[3800:,:]
Y1 = Y[0:3800]
Y2 = Y[3800:]

"""
model = build_model(X=M1, Y=Y1, hid_sizes=hid, rate=rate, lam=lam, it_max=it_max)
print "----------------------------------------------"
est = predict(model, M1)
T_err1 = np.sum(est!=Y1)/float(Y1.shape[0])
print "\tV1   Training Error = %.5f" %T_err1
est = predict(model, M2)
V_err1 = np.sum(est!=Y2)/float(Y2.shape[0])
print "\tV1 Validation Error = %.5f" %V_err1
print "----------------------------------------------"
M1 = M[200:,:]
M2 = M[0:200,:]
Y1 = Y[200:]
Y2 = Y[0:200]
model = build_model(X=M1, Y=Y1, hid_sizes=hid, rate=rate, lam=lam, it_max=it_max)
print "----------------------------------------------"
est = predict(model, M1)
T_err2 = np.sum(est!=Y1)/float(Y1.shape[0])
print "\tV2   Training Error = %.5f" %T_err2
est = predict(model, M2)
V_err2 = np.sum(est!=Y2)/float(Y2.shape[0])
print "\tV2 Validation Error = %.5f" %V_err2
print "=============================================="
print "      Average Training Error = %.5f" %((T_err1+T_err2)/2)
print "    Average Validation Error = %.5f" %((V_err1+V_err2)/2)
"""
model = build_model(X=M, Y=Y, hid_sizes=hid, rate=rate, lam=lam, it_max=it_max)

#-------------------Output-------------------
W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4']
f = open(sys.argv[2],'w')
f.write(str(hid[0])+','+str(hid[1])+','+str(hid[2])+'\n')
for i in range (W1.shape[0]):
    lis = ','.join(map(str, W1[i,:])) 
    f.write(lis+"\n")
for i in range (b1.shape[0]):
    lis = ','.join(map(str, b1[i,:])) 
    f.write(lis+"\n")
for i in range (W2.shape[0]):
    lis = ','.join(map(str, W2[i,:])) 
    f.write(lis+"\n")
for i in range (b2.shape[0]):
    lis = ','.join(map(str, b2[i,:])) 
    f.write(lis+"\n")
for i in range (W3.shape[0]):
    lis = ','.join(map(str, W3[i,:])) 
    f.write(lis+"\n")
for i in range (b3.shape[0]):
    lis = ','.join(map(str, b3[i,:])) 
    f.write(lis+"\n")
for i in range (W4.shape[0]):
    lis = ','.join(map(str, W4[i,:])) 
    f.write(lis+"\n")
for i in range (b4.shape[0]):
    lis = ','.join(map(str, b4[i,:])) 
    f.write(lis+"\n")
f.close()

