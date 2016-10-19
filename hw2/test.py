import sys
import numpy as np
#-------------------Acquire model-------------------
f = open(sys.argv[1],'r')
temp = f.read().splitlines()
raw=[]
for x in temp:
    raw.append(x.split(','))
f.close()
raw = np.array(raw)
raw[raw=="0."] = 0
raw = raw.astype(np.float)
w0 =raw[0];
w1_mat = np.matrix(raw[1:])
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
	M[:,i] = (M[:,i] - np.mean(M[:,i]))/np.std(M[:,i]);
# data is now in shape (4001,58)
M_mat = np.matrix(M)
z_mat = M_mat * w1_mat + w0
sig_mat = 1 / (1 + np.power(np.e,-z_mat))
est_mat = sig_mat
est_mat[est_mat>=0.5] = 1
est_mat[est_mat<0.5] = 0
est_mat = est_mat.astype(np.int)
#-------------------Output-------------------
f = open(sys.argv[3],'w')
f.write("id,label\n")
for i in range (est_mat.shape[0]):
	f.write(str(i+1)+","+str(np.sum(est_mat[i]))+"\n")
f.close()
