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
for i in range(0,4320,18): # starts from 1 because label[0] is already parsed above
	ft = raw[i:i+18,start_loc[1]+1:]
	ft = ft.astype(np.float)
	est.append(np.sum(w*ft) + b)
#-------------------Output-------------------
f = open('linear_regression.csv','w')
f.write("id,value\n")
for i in range (0,240,1):
	f.write("id_"+str(i)+","+str(est[i])+"\n")
f.close()

