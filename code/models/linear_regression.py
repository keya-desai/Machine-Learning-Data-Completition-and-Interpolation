import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def least_square_regression(X, Y):
    m, k = X.shape[0], X.shape[1]
    z = np.ones((m,1))
    X = np.append(X, z, axis=1)
    mat = np.linalg.inv(np.dot(np.transpose(X), X))
    w = np.dot(mat, np.dot(np.transpose(X), Y))
    return w

def ridge_regression(X, Y, l = 0.01):
    m, k = X.shape[0], X.shape[1]
    z = np.ones((m,1))
    X = np.append(X, z, axis=1)
    mat = np.linalg.inv(np.dot(np.transpose(X), X) + l*np.identity(k+1))
    w = np.dot(mat, np.dot(np.transpose(X), Y))
    return w

def predict(w, isFeatureReal, X_prime):
    X_prime = pd.DataFrame(X_prime)
    num_features = w.shape[0]
    for k in range(num_features-1):
        col_to_predict = k
        w_k = w[k]
        new_w = np.delete(w_k, col_to_predict)
        for idx, _ in isFeatureReal.loc[isFeatureReal[col_to_predict] == 0].iterrows():
            row = X_prime.iloc[idx]
            new_row = row.drop(col_to_predict)
            pred = - (np.dot(new_w, new_row) + w_k[-1])/w_k[col_to_predict]
            X_prime[col_to_predict].iloc[idx] = pred    
    return X_prime


def linear_regression_model(X_train, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test, ridge = False):

    if ridge:
    	w = ridge_regression(X_prime_train, X_prime_train, l = 0.01)
    else:
    	w = least_square_regression(X_prime_train, X_train)

    X_pred_train = predict(w, isFeatureReal_train, X_prime_train)
    X_pred_test = predict(w, isFeatureReal_test, X_prime_test)

    return X_pred_train, X_pred_test
