import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

def bernoulliNB(X ,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)
    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    scores = cross_val_score(bnb, X ,y, cv=14)
    return np.average(scores)

def decisiontree(X ,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    scores = cross_val_score(tree, X ,y, cv=14)

    return np.average(scores)

def knn(X ,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    scores = cross_val_score(knn_clf, X ,y, cv=14)
    return  np.average(scores)

def logreg(X ,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 24)

    clf = LogisticRegression(random_state = 0)
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X ,y, cv=14)

    return np.average(scores)

def gausnb(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    scores = cross_val_score(gnb, X, y, cv=14)

    return np.average(scores)

def svc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    sv = svm.SVC()
    sv.fit(X_train, y_train)

    scores = cross_val_score(sv, X, y, cv=14)

    return np.average(scores)


def rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(X_train, y_train)

    scores = cross_val_score(rf, X, y, cv=14)

    return np.average(scores)