from __future__ import division, print_function, unicode_literals
from learningrecognizer.preprocessing import globalVariables

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def trainandtest_split():

    df = pd.read_csv(globalVariables.dataFile_Path)

    train_set_full, test_set_full = train_test_split(df, test_size=0.2, random_state=42)
    print("Type of training_set/testing_set : ", type(train_set_full))
    print("Length of training_set and testing_Set : ", len(train_set_full), len(test_set_full), "\n")

    train_set = train_set_full.drop(["classLabel"], axis=1)
    X = train_set.as_matrix(columns=None)
    print("Type of X : ", type(X))
    print("Length of X : ", len(X), "\n")

    y = np.ravel(train_set_full.as_matrix(columns=["classLabel"]))
    print("Type of y : ", type(y))
    print("Length of y : ", len(y), "\n")


    test_set = test_set_full.drop(["classLabel"], axis=1)
    X_test = test_set.as_matrix(columns=None)

    y_test = test_set_full.as_matrix(columns=["classLabel"])

    return X, y, X_test, y_test


def trainSVM():

    X, y, X_test, y_test = trainandtest_split()

    model = SVC(kernel='linear', C=1, gamma=1)
    model.fit(X, y)

    n = model.score(X, y)
    print("ModelScore:", n)

    predictedLabels = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictedLabels)
    print("Accuracy:", accuracy)

    with open(globalVariables.SVM_Path, 'wb') as f:
        pickle.dump(model, f)

trainSVM()

