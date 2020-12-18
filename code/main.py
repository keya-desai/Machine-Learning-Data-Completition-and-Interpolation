from data import Data
from models.baseline import BaselineModel
import numpy as np
from models.sequential_prediction import model
import pandas as pd
from models.linear_regression import linear_regression_model
from models.knn import knnModel
from models.neural_network import NeuralNetwork
import matplotlib.pyplot as plt

def calculateError(train_true, train_pred, test_true, test_pred):
    
    diffTrain = (train_true - train_pred)**2
    diffTest = (test_true - test_pred)**2
    
    sumTrain = np.sum(diffTrain, axis = 1)
    sumTest = np.sum(diffTest, axis = 1)
    
    errTrain = np.sqrt(np.mean(sumTrain))
    errTest = np.sqrt(np.mean(sumTest))
    
    return errTrain, errTest

def calculateErrorOfCol(test_true, test_pred):
    
    diffTest = (test_true - test_pred)**2
    sumTest = np.sum(diffTest)
    errTest = np.sqrt(np.mean(sumTest))
    
    return errTest


def main():
	data = Data('../Data/')
	X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test = data.getData(newData = False)

	# Removing education num
	X_train = np.delete(X_train, 2, 1)
	X_test = np.delete(X_test, 2, 1)
	X_prime_train = np.delete(X_prime_train, 2, 1)
	X_prime_test = np.delete(X_prime_test, 2, 1)
	isFeatureReal_train = np.delete(isFeatureReal_train, 2, 1)
	isFeatureReal_test = np.delete(isFeatureReal_test, 2, 1)

	# Number of missing data points
	# print(np.sum(isFeatureReal_train == 0))

	print("\n***** Training data ***** ")
	print("X shape = ", X_train.shape)
	print("X prime shape = ", X_prime_train.shape)
	print("feature information shape = ", isFeatureReal_train.shape)

	print("\n***** Testing data ***** ")
	print("X shape = ", X_test.shape)
	print("X prime shape = ", X_prime_test.shape)
	print("Feature information shape = ", isFeatureReal_test.shape)

	categoricalFeatures = [[i - 1 for i in range(6,13)],
	             [i - 1 for i in range(13,29)],
	             [i - 1 for i in range(29,36)],
	             [i - 1 for i in range(36,50)],
	             [i - 1 for i in range(50,56)],
	             [i - 1 for i in range(56,61)],
	             [i - 1 for i in range(61,63)],
	             [i - 1 for i in range(63,104)],
	             [i - 1 for i in range(104,106)]]

	# Baseline model
	print("\n***** Baseline model *****")
	baseline_model = BaselineModel(X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test, categoricalFeatures)
	baseline_model.calculateSampleAverage()
	train_pred, test_pred = baseline_model.fillMissingValues()
	trainError, testError = calculateError(X_train, train_pred, X_test, test_pred)	
	print("Training Error = ", trainError)
	print("Testing Error = ", testError)


	# # Sequentical Prediction model
	print("\n***** Sequentical Prediction model *****")
	baseline_model_obj = BaselineModel(X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test, categoricalFeatures)
	train_pred, test_pred = model(X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test, categoricalFeatures)
	trainError, testError = calculateError(X_train, train_pred, X_test, test_pred)
	print("\nTraining Error = ", trainError)
	print("Testing Error = ", testError)


	print("\n***** Linear regression *****")
	train_pred, test_pred = linear_regression_model(X_train, X_prime_train, X_prime_test, pd.DataFrame(isFeatureReal_train), pd.DataFrame(isFeatureReal_test))
	trainError, testError = calculateError(X_train, X_train, X_test, test_pred)
	print("\nTraining Error = ", trainError)
	print("Testing Error = ", testError)

	# Are there some features it is really good at predicting and some it is really poor at predicting? Why do you think that is?
	# np.save('../Data/' + 'Prediction/test.npy', test_pred)
	# test_pred = np.load('../Data/' + 'Prediction/test.npy', allow_pickle = True)
	# _, testError = calculateError(X_train, X_train, X_test, test_pred)
	# print(testError)

	# k = len(test_pred[0])
	# totalError = 0
	# for idx in range(k):
	# 	testError = calculateErrorOfCol(X_test[:, idx], test_pred[:, idx])
	# 	# if idx not in [2, 105]:
	# 	totalError += testError 
	# 	print(idx, testError)
	# print("totalError = ", totalError)



	print("\n***** Ridge regression *****")
	train_pred, test_pred = linear_regression_model(X_train, X_prime_train, X_prime_test, pd.DataFrame(isFeatureReal_train), pd.DataFrame(isFeatureReal_test), ridge = True)
	trainError, testError = calculateError(X_train, train_pred, X_test, test_pred)
	print("\nTraining Error = ", trainError)
	print("Testing Error = ", testError)


	print("\n ***** KNN Model *****")
	knn_model = knnModel(pd.DataFrame(X_prime_train), pd.DataFrame(isFeatureReal_train), pd.DataFrame(X_prime_test), pd.DataFrame(isFeatureReal_test), categoricalFeatures, n = 5)
	knn_model.predictData()
	testErr = knn_model.calculateTestError(X_test[0:5,:])
	print("\n Test Error:", testErr)


	# Plotting for different n
	num_neighbors = [10, 25, 50, 100, 500]
	test_err = []

	for num in num_neighbors:
	    print("Number of nearest neighbors:", num)
	    knn_model = knnModel(pd.DataFrame(X_prime_train), pd.DataFrame(isFeatureReal_train), pd.DataFrame(X_prime_test), pd.DataFrame(isFeatureReal_test), categoricalFeatures, n = num)
	    knn_model.predictData()
	    testErr = knn_model.calculateTestError(X_test[0:5, :])
	    test_err.append(testErr)
	    print("Testing Error:", testErr)
	    print("**************************")

	plt.figure(figsize=(8,6))
	plt.plot(num_neighbors, test_err, marker='x')
	plt.title("Testing error for different number of Nearest Neighbors")
	plt.xlabel("Number of neighbors (k)")
	plt.ylabel("Testing error")
	plt.show()



	print("\n ***** Neural Network *****")
	trainX = X_prime_train.T
	trainY = X_train.T
	valX = X_prime_train.T
	valY = X_train.T
	valFeatureInfo = isFeatureReal_train.T
	testX = X_prime_test.T
	testY = X_test.T

	nn = NeuralNetwork(trainX, trainY, valX, valY, valFeatureInfo, num_hidden= 1, epochs= 100, learning_rate=0.05, num_nodes_layers=[10],
	                       activation_function="sigmoid", batch_size = 1)
	trainLossArr, valLossArr = nn.model()

	pred = nn.predict(testX, isFeatureReal_test.T)
	testErr = np.sqrt(np.mean(np.sum((pred - testY) ** 2, axis = 1)))
	print("\nTesting error:", testErr)

	# PLot training error
	num_epochs = [i for i in range(len(trainLossArr))]
	plt.figure(figsize=(8,6))
	plt.plot(num_epochs, trainLossArr, marker='x')
	plt.title("Training loss v/s Epochs")
	plt.xlabel("Number of epochs")
	plt.ylabel("Training Loss")
	plt.show()

	# Plotting validation error
	plt.figure(figsize=(8,6))
	plt.plot(num_epochs, valLossArr, marker='x')
	plt.title("Validation loss v/s Epochs")
	plt.xlabel("Number of epochs")
	plt.ylabel("Validation Loss")
	plt.show()


if __name__ == "__main__":
	main()