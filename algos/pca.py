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



# Variance for each component
def pca_fun(X):
    pca_values=[]
    for i in range(2,130):
        pca_values.append(PCA(n_components=i).fit(X).explained_variance_ratio_.sum())

    #plt.bar(range(len(pca_values)), [imp for imp in pca_values], align='center')
    return (len(pca_values),pca_values)