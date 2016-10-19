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
w0 = 1
w1_mat = np.ones((57,1))
it_max = 100000 # iterations
rate = 0.1 # learning rate (fixed)
lam = 50 # lambda for regularization
ada_w0 = 0
ada_w1 = np.zeros((57,1))
eps = 1e-6
w0_grad = 0
w1_grad = np.zeros((57,1))

for it in range(it_max):
	z_mat = M_mat * w1_mat + w0
	sig_mat = 1 / (1 + np.power(np.e,-z_mat))
	err_mat = Y_mat - sig_mat;
	if it%10000 == 0:
		est_mat = sig_mat
		est_mat[est_mat>=0.5] = 1
		est_mat[est_mat<0.5] = 0
		print "[",it,"/",it_max,"]\t error = ",np.sum(Y_mat != est_mat)/4001.0
	# Gradient Descent
	w0_grad = -w0 * np.sum(err_mat)
	w1_grad = -M_mat.T * err_mat # linear weights
	# Regularization
	w1_grad = w1_grad + 2*lam*np.array(w1_mat)
	# Adagrad
	ada_w0 = np.sqrt(np.square(ada_w0) + np.square(w0_grad))
	ada_w1 = np.sqrt(np.square(ada_w1) + np.square(w1_grad))
	# Update Step
	w0 = w0 - rate * w0_grad / (ada_w0 + eps)
	w1_mat = w1_mat - rate * w1_grad / (ada_w1 + eps)
#-------------------Output-------------------
f = open(sys.argv[2],'w')
f.write(str(w0)+"\n")
for i in range (57):
    f.write(str(np.sum(w1_mat[i]))+"\n")
f.close()
