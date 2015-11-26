# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 23:24:05 2015

@author: Naman
"""
import scipy.io
#from sklearn import svm,grid_search
import numpy as np
from sklearn import preprocessing
#from sklearn.lda import LDA
from sklearn import manifold


mat = scipy.io.loadmat('filename.mat')

test=mat.get('test');
train=mat.get('train');
train_label=mat.get('train_label');
train_label=(np.array(train_label)).ravel();
train=np.array(train);

scale1 = preprocessing.StandardScaler().fit(train)
train_scaled=scale1.transform(train)

model=manifold.TSNE()
Y=model.fit_transform(train_scaled[0:50000])
np.save('tsneset',Y)
Y0=[]
Y1=[]
for i in range(50000):
    if train_label[i]==0:
        Y0.append([Y[i,0],Y[i,1]])
    if train_label[i]==1:
        Y1.append([Y[i,0],Y[i,1]])

Y0=np.array(Y0);
Y1=np.array(Y1);

#plt.scatter(Y0[:, 0], Y0[:, 1],color='red'),plt.scatter(Y1[:, 0], Y1[:, 1])
 

#param_grid = [
#  {'C': [0.01], 'kernel': ['linear']},
## {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],  'gamma': [1, 0.01, 0.001, 0.0001,0.05,0.005, 0.0005], 'kernel': ['rbf']},
## {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],  'degree' :[2, 3, 4], 'kernel': ['poly']},
# ]
##
#svr = svm.SVC()
#svrsearch= grid_search.GridSearchCV(svr,param_grid,cv=10)
#y_rbf = svrsearch.fit(train, train_label)
#
#predict=[]
#a=[]
#
#for i in range(50000):
#    predict.append(y_rbf.predict(train[i]))
#    a.append(i)
