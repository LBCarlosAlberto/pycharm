#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:50:47 2017

@author: nz
"""
import numpy as np
import pickle
from scipy import io

from sklearn.linear_model import SGDClassifier
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn import metrics
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Analyzer import evaluate_and_report, plot_confusion_matrix
topics = ['business','environment','fashion','lifeandstyle',\
                'politics','sport','technology','travel','world']

train = io.mmread('1gram_train.mtx')
train_tags = pickle.load(open('1gram_train_tags.pk', 'rb'))
test = io.mmread('1gram_test.mtx')
test_tags = pickle.load(open('1gram_test_tags.pk', 'rb'))
'''
svd = TruncatedSVD(n_components=5000)
svd.fit(train)
xtrain_svd = svd.transform(train)
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
'''
#self-made grid search
components = [5000]
penalties = ['l2', 'elasticnet']
class_weights = [None, 'balanced']
best_score = 0
best_model = None
best_train = None
best_test = None
best_n_components = None
for component in components:
    svd = TruncatedSVD(n_components = component)
    #train
    xtrain_svd = svd.fit_transform(train)
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    #test
    xtest_svd = svd.transform(test)
    scl = preprocessing.StandardScaler()
    scl.fit(xtest_svd)
    xtest_svd_scl = scl.transform(xtest_svd)
    for penalty in penalties:
        for class_weight in class_weights:
            sgd_clf = SGDClassifier(loss='hinge', penalty=penalty, \
                    alpha=0.0001, l1_ratio=0.15, \
                    fit_intercept=True, max_iter=None, \
                    tol=None, shuffle=True, verbose=0, \
                    epsilon=0.1, n_jobs=-1, random_state=None, \
                    learning_rate='optimal', eta0=0.0, power_t=0.5, \
                    class_weight=class_weight, warm_start=False, \
                    average=False, n_iter=None)
            scores = cross_val_score(sgd_clf, xtrain_svd_scl, train_tags, scoring = 'f1_macro')
            print(scores)
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_model = sgd_clf
                best_train = xtrain_svd_scl
                best_test = xtest_svd_scl
                best_n_components = component


best_model.fit(xtrain_svd_scl, train_tags)
prediction = evaluate_and_report(best_model, xtest_svd_scl, test_tags, topics)
# Compute confusion matrix
cnf_matrix = confusion_matrix(test_tags, prediction)
np.set_printoptions(precision=2)
model_file = 'train_model_svm.sav'
pickle.dump(best_model, open(model_file, 'wb'))
# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=topics,
                  title='Confusion matrix, without normalization')
#plt.show()
#image_name = 'confusion_matrix_' + model + '.png'
#plt.savefig(image_name)

            
            





'''
sgd_clf = SGDClassifier(loss='hinge', penalty='l2', \
                    alpha=0.0001, l1_ratio=0.15, \
                    fit_intercept=True, max_iter=None, \
                    tol=None, shuffle=True, verbose=0, \
                    epsilon=0.1, n_jobs=-1, random_state=None, \
                    learning_rate='optimal', eta0=0.0, power_t=0.5, \
                    class_weight=None, warm_start=False, \
                    average=False, n_iter=None)

svd = TruncatedSVD()
scl = preprocessing.StandardScaler()
sgd_clf = SGDClassifier()

clf = Pipeline([('svd', svd),
                ('scl', scl),
                ('clf', sgd_clf)])

param_grid = {'svd__n_components' : [500, 1000],
              'clf__loss' : ['hinge'],
              'clf__penalty' : ['l2', 'elasticnet'],
              'clf__class_weight' : [None, 'balanced']}
model = GridSearchCV(estimator = clf, param_grid = param_grid, scoring = 'f1_macro',\
                     verbose = 0, n_jobs = -1, iid = True, refit = True, cv = 5)

model = SGDClassifier(loss='hinge', penalty='l2', \
                    alpha=0.0001, l1_ratio=0.15, \
                    fit_intercept=True, max_iter=None, \
                    tol=None, shuffle=True, verbose=0, \
                    epsilon=0.1, n_jobs=-1, random_state=None, \
                    learning_rate='optimal', eta0=0.0, power_t=0.5, \
                    class_weight=None, warm_start=False, \
                    average=False, n_iter=None)
model.fit(xtrain_svd_scl, train_tags)



xtest_svd = svd.transform(test)
scl = preprocessing.StandardScaler()
scl.fit(xtest_svd)
xtest_svd_scl = scl.transform(xtest_svd)


prediction = evaluate_and_report(model, xtest_svd_scl, test_tags, topics)
# Compute confusion matrix
cnf_matrix = confusion_matrix(test_tags, prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, classes=topics,
                  title='Confusion matrix, without normalization')
#plt.show()
#image_name = 'confusion_matrix_' + model + '.png'
#plt.savefig(image_name)

'''