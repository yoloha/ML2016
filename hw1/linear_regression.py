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
# Finding the first AMB_TEMP to parse the file
find_loc = np.where(raw=='AMB_TEMP')
start_loc = np.array([find_loc[0][0],find_loc[1][0]])
# Replace NR with 0
raw[raw=='NR'] = 0
# Find the set of labels (such as AMB_TEMP)
#**** this code is currently fixed for 18 labels ******** 
label = raw[start_loc[0]:(start_loc[0]+18),start_loc[1]]
data = raw[raw[:,start_loc[1]]==label[0]]
data = data[:,start_loc[1]+1:]
data = data.astype(np.float)
data = data.reshape(1,-1)
# further concatenate data with other sets of data corresponding to each label
for i in range(1,label.size,1): # starts from 1 because label[0] is already parsed above
    temp = raw[raw[:,start_loc[1]]==label[i]]
    temp = temp[:,start_loc[1]+1:]
    temp = temp.astype(np.float)
    temp = temp.reshape(1,-1)
    data = np.vstack((data,temp))
# data[:,:] is now a 2-d array that correspond to each sample
# each row correspond to each label
# each column correspond to the consecutive sample time
<<<<<<< HEAD
# Then we'll have to eat data in chunks, 9 samples as a set of features, then predict the (9+1)th set's PM2.5
# Each row of M consists of 163 parameters, with a total of 5751 rows
M = np.zeros((5652,163))# 1 offset + 162 linear weights
Y = np.zeros((5652,1))# response to each solution
for month in range(12):
    for i in range(9,471,1):
        temp = data[:,month*480+i-9:month*480+i]
        temp = temp.reshape(1,-1)
        M[month*471+i-9,0] = 1
        M[month*471+i-9,1:163] = temp
        Y[month*471+i-9] = data[9,month*480+i]
M_mat = np.matrix(M)
Y_mat = np.matrix(Y)
opt_mat = (M_mat.T*M_mat).I*M_mat.T*Y_mat
# After the calculation of the optimal matrix solution, M_mat is stripped into 162 columns
M_mat = M_mat[:,1:]
w0 = opt_mat[0]
w1_mat = opt_mat[1:]
it_max = 50000 # iterations
rate = 0.01 # learning rate (fixed)
lam = 1000 # lambda for regularization
ada_w0 = 0
ada_w1 = np.zeros((162,1))
eps = 1e-6
w0_grad = 0
w1_grad = np.zeros((162,1))
for it in range(it_max):
    est_mat = w0 + M_mat * w1_mat
    err_mat = Y_mat - est_mat
    if it%100==0:
        print "[",it,"/",it_max,"]\t",np.sqrt(np.sum(np.array(err_mat)**2)/5652)
    # Gradient Descent
    w0_grad = 2*np.sum(err_mat) # linear offset
    w1_grad = (-2*M_mat.T * err_mat) # linear weights
    # Regularization
    w1_grad = w1_grad + 2*lam*np.array(w1_mat)
    # Adagrad
    ada_w0 = np.sqrt(ada_w0**2 + w0_grad**2)
    ada_w1 = np.sqrt(np.square(ada_w1) + np.square(w1_grad))
    # Update Step
    w0 = w0 - rate*w0_grad/(ada_w0+eps)
    w1_mat = w1_mat - rate*w1_grad/(ada_w1+eps)
=======
# Then we'll have to eat data in chunks, 9 samples as a set of features, then predict the (9+1)th set's PM2.5 
b = 20
w = np.zeros((18,9))
it_max = 100000 # iterations
rate = 0.1 # learning rate (fixed)
lam = 1 # lambda for regularization
ada_w = np.zeros((18,9))
ada_b = 0
for it in range(0,it_max,1):
	sum_sq_err = 0 # sum of squared errors
	w_grad = np.zeros((18,9))
	b_grad = 0
	for i in range(9,data.shape[1],1): # i = 0~5759
		ft = data[:,i-9:i]
		est = np.sum(w*ft) + b
		# PM2.5 data is on row index 9, i.e data[9,:]
		err = data[9,i] - est
		sum_sq_err = sum_sq_err + err**2
		# Gradient descent
		w_grad = w_grad + 2*err*(-ft)
		b_grad = b_grad + 2*err
		# Regularization
		w_grad = w_grad + 2*lam*w
		b_grad = b_grad + 2*lam*b
	ada_w = np.sqrt(ada_w**2 + w_grad**2)
	ada_b = np.sqrt(ada_b**2 + b_grad**2)
	w = w - rate*w_grad/(ada_w+0.001)
	b = b - rate*b_grad/(ada_b+0.001)
	print "[",it,"/",it_max,"]\t",sum_sq_err
print err
>>>>>>> 5f6b1fd9445ea09e86e0b582b2200bad6db99121
#-------------------Test-------------------
f = open(sys.argv[2],'r')
temp = f.read().splitlines()
raw=[]
for x in temp:
    raw.append(x.split(','))
f.close()
raw = np.array(raw)
# Finding the first AMB_TEMP to parse the file
find_loc = np.where(raw=='AMB_TEMP')
start_loc = np.array([find_loc[0][0],find_loc[1][0]])
# Replace NR with 0
raw[raw=='NR'] = 0
est = []
M = np.zeros((1,163))
for i in range(0,4320,18): # starts from 1 because label[0] is already parsed above
    M = raw[i:i+18,start_loc[1]+1:]
    M = M.astype(np.float)
    M = M.reshape((1,-1))
    pred = np.sum(M*w1_mat + w0) # sum is used here to remove brackets (matrix->value)
    if pred < 0:
    	pred = 0.0
    est.append(pred)
#-------------------Output-------------------
f = open('linear_regression.csv','w')
f.write("id,value\n")
for i in range (0,240,1):
    f.write("id_"+str(i)+","+str(est[i])+"\n")
f.close()

