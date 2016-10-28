import sys
import numpy as np
#-------------------Train-------------------
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
# data is now in shape (4001,58)
M_mat = np.matrix(M)
#M_mat = np.insert(M_mat,0,values=0,axis=1)
Y_mat = np.matrix(Y).T
b = 1
W = np.ones((57,1))
it_max = 10000 # iterations
rate = 0.1 # learning rate (fixed)
lam = 0 # lambda for regularization
ada_b = 0
ada_w1 = np.zeros((57,1))
eps = 1e-6
db = 0
dW = np.zeros((57,1))
for it in range(it_max):
	z_mat = M_mat * W + b
	sig_mat = 1 / (1 + np.power(np.e,-z_mat))
	err_mat = Y_mat - sig_mat;
	if it%1000 == 0:
		est_mat = sig_mat
		est_mat[est_mat>=0.5] = 1
		est_mat[est_mat<0.5] = 0
		print "[%5d"%it,"/",it_max,"] loss = %12.5f"%np.sum(-err_mat),"  error = %.3f"%(np.sum(Y_mat != est_mat)/4001.0)
	# Gradient Descent
	db = -1.0 * np.sum(err_mat)
	dW = -M_mat.T * err_mat # linear weights
	# Regularization
	dW = dW + 2*lam*np.array(W)
	# Adagrad
	ada_b = np.sqrt(np.square(ada_b) + np.square(db))
	ada_w1 = np.sqrt(np.square(ada_w1) + np.square(dW))
	# Update Step
	b = b - rate * db / (ada_b + eps)
	W = W - rate * dW / (ada_w1 + eps)
#-------------------Output-------------------
f = open(sys.argv[2],'w')
f.write(str(b)+"\n")
for i in range (57):
    f.write(str(np.sum(W[i]))+"\n")
f.close()
def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))