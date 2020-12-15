# from models.baseline import BaselineModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linear_regression_model(X, y, learning_rate, epochs):
    # weight initialization
    m, k = X.shape
    w = np.random.rand(k)
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
            print("Breaking at epoch = ", e)
            break

        l_prev = l
        loss.append(np.mean(l))

    print("Final loss = ", l)

    # plt.plot(loss)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.show()

    # print(b, w)

    return w, b




def dev_model(X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test, baseline_model_obj):

    # Trying out with missing age column 
    col_to_predict = 100
    nFeatures = len(X_prime_train[0])
    # List of columns index to fill using baesline methods
    col_idx_list = [i for i in range(nFeatures) if i != col_to_predict]

    baseline_model_obj.calculateSampleAverage()
    X_prime_train, X_prime_test = baseline_model_obj.fillMissingValues(col_idx_list)


    # Filtering rows where col_to_predict is missing
    m, k = X_prime_train.shape
    isFeatureReal_train_temp = np.ones((m, k))
    isFeatureReal_train_temp[:, col_to_predict] = isFeatureReal_train[:, col_to_predict]
    # isFeatureReal_train_temp[:, 1:] = 1
    x_training = X_prime_train.where(isFeatureReal_train_temp > 0)
    x_training.dropna(inplace = True)
    # Sanity check
    # print(len(isFeatureReal_train_temp[isFeatureReal_train_temp[:, col_to_predict] == 1]))
    # print(x_training.shape)

    print(x_training.head())

    m_training, k_training = x_training.shape
    dummy_y = np.array([0] * m_training)
    w, b = linear_regression_model(np.array(x_training), dummy_y, learning_rate = 0.00001, epochs = 10000)

    isFeatureReal_train = pd.DataFrame(isFeatureReal_train)
    isFeatureReal_test = pd.DataFrame(isFeatureReal_test)


    print(np.mean(np.matmul(np.array(x_training), w) + b))

    new_w = np.delete(w, col_to_predict)
    cnt = 0
    for idx, _ in isFeatureReal_train.loc[isFeatureReal_train[col_to_predict] == 0].iterrows(): 
        row = X_prime_train.iloc[idx]
        new_row = row.drop(col_to_predict)
        pred = - (np.dot(new_w, new_row) + b )/w[col_to_predict]
        print(idx, pred)
        X_prime_train[col_to_predict].iloc[idx] = pred
        cnt += 1
        if cnt > 5 : break


    new_w = np.delete(w, col_to_predict)
    for idx, _ in isFeatureReal_test.loc[isFeatureReal_test[col_to_predict] == 0].iterrows():
        row = X_prime_test.iloc[idx]
        new_row = row.drop(col_to_predict)
        pred = - (np.dot(new_w, new_row) + b )/w[col_to_predict]
        X_prime_test[col_to_predict].iloc[idx] = pred    

    return X_prime_train, X_prime_test