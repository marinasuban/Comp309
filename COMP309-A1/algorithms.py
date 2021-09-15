from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sklearn
from sklearn.utils._testing import ignore_warnings


def Process(data):
    x = data[0]
    x = StandardScaler().fit_transform(x)
    return (x, data[1])

def KNN(data, k):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1].ravel(), test_size=0.5)  # 50% Test Data
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def Gaussian(data, k):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1].ravel(), test_size=0.5)
    gaussianclf = GaussianNB(var_smoothing=k)
    gaussianclf.fit(X_train, y_train)
    y_pred = gaussianclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def LogRegression(data, k):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1].ravel(), test_size=0.5)
    regressionclf = LogisticRegression(C=k)
    regressionclf.fit(X_train, y_train)
    y_pred = regressionclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def DT(data, k):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1].ravel(), test_size=0.5)
    treeclf = DecisionTreeClassifier(max_depth=k)
    treeclf.fit(X_train, y_train)
    y_pred = treeclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def GradientDT(data, k):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1].ravel(), test_size=0.5)
    gradientclf = GradientBoostingClassifier(max_depth=k)
    gradientclf.fit(X_train, y_train)
    y_pred = gradientclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def Forest(data, k):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1].ravel(), test_size=0.5)  # 50% Test Data
    forestclf = RandomForestClassifier(max_depth=k)
    forestclf.fit(X_train, y_train)
    y_pred = forestclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

@ignore_warnings(category=ConvergenceWarning)
def MLP(data, k):
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1].ravel(), test_size=0.5)
    mlpclf = MLPClassifier(alpha=k)
    mlpclf.fit(X_train, y_train)
    y_pred = mlpclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def GaussianPart2(data):

    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1].ravel(), test_size=0.95)
    gaussianclf = GaussianNB()
    gaussianclf.fit(X_train, y_train)
    y_pred = gaussianclf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy