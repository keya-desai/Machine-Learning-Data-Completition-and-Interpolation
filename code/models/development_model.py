# from models.baseline import BaselineModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linear_regression_model(X, y, learning_rate, epochs):
    # weight initialization
    m, k = X.shape
    w = np.random.randn(k)
    b = 1
    loss = []
    l_prev = 0
    # print(X.shape, w.shape, y.shape)
    for e in range(epochs):
        y_pred = np.matmul(X, w) + b

        b_gradient = -2 * np.sum(y - y_pred) / m
        w_gradient = -2 * np.sum(np.matmul((y - y_pred), X)) / m

        w -= learning_rate * w_gradient
        b -= learning_rate * b_gradient

        l = np.mean((y - np.matmul(X, w) - b)**2)

        if abs(l_prev - l) < 0.00001:
            # print("Breaking at epoch = ", e)
            break

        l_prev = l
        loss.append(np.mean(l))

    # print("Final loss = ", l)
    return w, b

def softmax(X):
    X -= np.max(X)
    sum_ = np.sum(np.exp(X), axis = 1)
    return np.exp(X)/sum_[:, None]

def logistic_regression_model(X, Y, learning_rate, epochs):

    # weight initialization
    m, k = X.shape
    _, c = Y.shape
    bias = np.array([1] * c).reshape(1, c)
    W = np.random.normal(0, 0.01, (k, c))
    W = np.append(bias, W, axis = 0)
    # print("Weight matrix : ", W.shape)
    # b = 1
    loss = []
    # print(X.shape)
    bias_term = np.array([1] * m).reshape(m, 1)
    X = np.append(bias_term, X, axis = 1)
    # print("Input matrix : ", X.shape)
    # print("Output matrix : ", Y.shape)

    for _ in range(epochs):

        Z = np.matmul(X, W)
        # Z = (m x c)
        # Computing softmax
        Z_softmax = softmax(Z)
        # print("Z = XW : ", Z_softmax.shape)

        dZ = Y - Z_softmax
        # print("DZ = ", dZ.shape)
        dW = np.matmul(X.T, dZ)/m
        # print("dW : ", dW.shape)
        W -= learning_rate * dW

        l = - np.sum(np.multiply(Y, np.log(Z_softmax)))/m

    # print("Final loss = ", l)
    return W

def predict(x, W):
    
    x = np.append(1, x)
    x = x.reshape(1, -1)

    _, c = W.shape
    z = np.matmul(x, W)
    z_softmax = softmax(z)
    pred = np.array([0.] * c)
    maxIdx = np.argmax(z_softmax)
    pred[maxIdx] = 1.

    return pred

def percent_data_missing(df):

    df = pd.DataFrame(df)
    # print(np.sum(df == 0))
    # print(np.argsort(np.sum(df == 0)))
    sorted_index_list  = np.argsort(np.sum(df == 0))
    return sorted_index_list


def ensemble_model_trial(X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test, baseline_model_obj, categoricalFeatures):

    sorted_index_list =  percent_data_missing(isFeatureReal_train)
    _, k = X_prime_train.shape

    # Fill with baseline values
    baseline_model_obj.calculateSampleAverage()
    train_baseline, test_baseline = baseline_model_obj.fillMissingValues()


    categoricalFeaturesFlatten = [ele for sublist in categoricalFeatures for ele in sublist]
    # print(categoricalFeaturesFlatten)

    index_column_to_predict = 0
    predicted_indexes = []

    for index_column_to_predict in sorted_index_list:
        if index_column_to_predict not in categoricalFeaturesFlatten:
            X_prime_train[:, index_column_to_predict], X_prime_test[:, index_column_to_predict] = predictContinuousFeature(pd.DataFrame(X_prime_train), pd.DataFrame(X_prime_test), isFeatureReal_train, isFeatureReal_test, index_column_to_predict)    
        else:
            if index_column_to_predict in predicted_indexes:
                continue

            for i, subList in enumerate(categoricalFeatures):
                if index_column_to_predict in subList:
                    start = subList[0]
                    end = subList[-1]
                    indexInCategoricalList = i
                    break

            X_prime_train[:, start : end + 1], X_prime_test[:, start : end + 1] = predictCategoricalFeature(pd.DataFrame(X_prime_train), pd.DataFrame(X_prime_test), isFeatureReal_train, isFeatureReal_test, categoricalFeatures, index_column_to_predict, start, end, indexInCategoricalList)

            for _ in range(start, end + 1):
                predicted_indexes.append(_)

        print(index_column_to_predict)


    return X_prime_train, X_prime_test

def ensemble_model(X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test, baseline_model_obj, categoricalFeatures):

    _, k = X_prime_train.shape

    # Fill with baseline values
    baseline_model_obj.calculateSampleAverage()
    train_baseline, test_baseline = baseline_model_obj.fillMissingValues()


    categoricalFeaturesFlatten = [ele for sublist in categoricalFeatures for ele in sublist]
    # print(categoricalFeaturesFlatten)

    index_column_to_predict = 0
    while index_column_to_predict < k:
        if index_column_to_predict not in categoricalFeaturesFlatten:
            X_prime_train[:, index_column_to_predict], X_prime_test[:, index_column_to_predict] = predictContinuousFeature(train_data, test_data, isFeatureReal_train, isFeatureReal_test, index_column_to_predict)    
            index_column_to_predict += 1
            # pass

        else:
            # break
            for i, subList in enumerate(categoricalFeatures):
                if index_column_to_predict in subList:
                    start = subList[0]
                    end = subList[-1]
                    indexInCategoricalList = i
                    break

            X_prime_train[:, start : end + 1], X_prime_test[:, start : end + 1] = predictCategoricalFeature(train_data, test_data, isFeatureReal_train, isFeatureReal_test, categoricalFeatures, index_column_to_predict, start, end, indexInCategoricalList)
            index_column_to_predict += end - start + 1

        print(index_column_to_predict)


    return X_prime_train, X_prime_test



def predictContinuousFeature(train_data, test_data, isFeatureReal_train, isFeatureReal_test, col_to_predict):

    # Trying out with missing age column 
    # col_to_predict = 100
    nFeatures = len(train_data[0])
    # # List of columns index to fill using baesline methods
    # col_idx_list = [i for i in range(nFeatures) if i != col_to_predict]

    # baseline_model_obj.calculateSampleAverage()
    # train_data, test_data = baseline_model_obj.fillMissingValues(col_idx_list)

    # Filtering rows where col_to_predict is missing
    m, k = train_data.shape
    isFeatureReal_train_temp = np.ones((m, k))
    isFeatureReal_train_temp[:, col_to_predict] = isFeatureReal_train[:, col_to_predict]
    x_training = train_data.where(isFeatureReal_train_temp > 0)
    x_training.dropna(inplace = True)
    # Sanity check
    # print(len(isFeatureReal_train_temp[isFeatureReal_train_temp[:, col_to_predict] == 1]))
    # print(x_training.shape)

    m_training, k_training = x_training.shape
    isFeatureReal_train = pd.DataFrame(isFeatureReal_train)
    isFeatureReal_test = pd.DataFrame(isFeatureReal_test)

    dummy_y = np.array([0] * m_training)
    w, b = linear_regression_model(np.array(x_training), dummy_y, learning_rate = 0.00001, epochs = 100)

    new_w = np.delete(w, col_to_predict)
    cnt = 0

    for idx, _ in isFeatureReal_train.loc[isFeatureReal_train[col_to_predict] == 0].iterrows(): 
        row = train_data.iloc[idx]
        new_row = row.drop(col_to_predict)
        pred = - (np.dot(new_w, new_row) + b )/w[col_to_predict]
        # print(idx, pred)
        train_data[col_to_predict].iloc[idx] = pred
        cnt += 1
        # if cnt > 5 : break

    for idx, _ in isFeatureReal_test.loc[isFeatureReal_test[col_to_predict] == 0].iterrows():
        row = test_data.iloc[idx]
        new_row = row.drop(col_to_predict)
        pred = -(np.dot(new_w, new_row) + b )/w[col_to_predict]
        test_data[col_to_predict].iloc[idx] = pred    

    return train_data[col_to_predict], test_data[col_to_predict]


def predictCategoricalFeature(train_data, test_data, isFeatureReal_train, isFeatureReal_test, categoricalFeatures, col_to_predict, start, end, indexInCategoricalList):

    # Filtering rows where col_to_predict is missing
    m, k = train_data.shape
    isFeatureReal_train_temp = np.ones((m, k))
    isFeatureReal_train_temp[:, col_to_predict] = isFeatureReal_train[:, col_to_predict]
    x_training = train_data.where(isFeatureReal_train_temp > 0)
    x_training.dropna(inplace = True)

    # Sanity check
    # print(len(isFeatureReal_train_temp[isFeatureReal_train_temp[:, col_to_predict] == 1]))
    # print(x_training.shape)

    isFeatureReal_train = pd.DataFrame(isFeatureReal_train)
    isFeatureReal_test = pd.DataFrame(isFeatureReal_test)

    categoricalFeaturesFlatten = [ele for sublist in categoricalFeatures for ele in sublist]

    if col_to_predict in categoricalFeaturesFlatten:
        for i, subList in enumerate(categoricalFeatures):
            if col_to_predict in subList:
                start = subList[0]
                end = subList[-1]
                indexInCategoricalList = i
                break

        y = x_training.iloc[:, start:end+1]
        X = x_training.drop(categoricalFeatures[indexInCategoricalList], axis = 1)
        W = logistic_regression_model(np.array(X), np.array(y), learning_rate = 0.0001, epochs = 100)
        
        # pred = predict(X[0], W)
        # print(pred)
        # print(y[0])

        for idx, _ in isFeatureReal_train.loc[isFeatureReal_train[col_to_predict] == 0].iterrows(): 
            row = train_data.iloc[idx]
            x = row.drop(categoricalFeatures[indexInCategoricalList])
            pred = predict(x, W)
            train_data.iloc[idx, start : end + 1] = pred

        for idx, _ in isFeatureReal_test.loc[isFeatureReal_test[col_to_predict] == 0].iterrows():
            row = test_data.iloc[idx]
            x = row.drop(categoricalFeatures[indexInCategoricalList])
            pred = predict(x, W)
            test_data.iloc[idx, start : end + 1] = pred

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # print(train_data[:, start : end + 1].shape)
    return train_data[:, start : end + 1], test_data[:, start : end + 1]


