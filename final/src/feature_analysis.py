import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif
from numpy import *
#----- Read Data -----
with open('train2_p_nodup') as f:
    raw = f.read().splitlines()
data_tr = []
for line in raw:
    data_tr.append(line.split(','))
data_tr = np.array(data_tr)
data_tr = data_tr.astype(float)


X = data_tr[:,:-1]
Y = data_tr[:,-1]

X_indices = np.arange(X.shape[-1])

selector = SelectPercentile(f_classif, percentile=100)
selector.fit(X, Y)
scores = -np.log10(selector.pvalues_ + 10**-10)
scores[isnan(scores)] = 0 
scores /= scores.max()

plt.figure(figsize=(10,10))
locs, labels = plt.xticks()
plt.title("CSAD features' ANOVA f-test score")
plt.setp(labels, rotation=45)
plt.bar(X_indices - .25, scores, width=.5,
        color='darkorange')

plt.subplots_adjust(top=0.95, bottom=0.3)

plt.show()