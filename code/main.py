from data import Data
from models.baseline import BaselineModel
import numpy as np
from models.development_model import dev_model
import pandas as pd

def calculateError(train_true, train_pred, test_true, test_pred):
    
    m, k = train_pred.shape
    m_test, k_test = test_pred.shape
    
    diffTrain = (train_true - train_pred)**2
    diffTest = (test_true - test_pred)**2
    
    sumTrain = np.sum(diffTrain, axis = 1)
    sumTest = np.sum(diffTest, axis = 1)
    
    errTrain = np.mean(np.sqrt(sumTrain))
    errTest = np.mean(np.sqrt(sumTest))
    
    return errTrain, errTest


def main():
	data = Data('../Data/')
	X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test = data.getData(newData = False)

	print("\n***** Training data ***** ")
	print("X shape = ", X_train.shape)
	print("X prime shape = ", X_prime_train.shape)
	print("feature information shape = ", isFeatureReal_train.shape)

	print("\n***** Testing data ***** ")
	print("X shape = ", X_test.shape)
	print("X prime shape = ", X_prime_test.shape)
	print("Feature information shape = ", isFeatureReal_test.shape)

	# exit()
	# Baseline model
	# baseline_model = BaselineModel(X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test)
	# baseline_model.calculateSampleAverage()
	# train_pred, test_pred = baseline_model.fillMissingValues()
	# trainError, testError = calculateError(X_train, train_pred, X_test, test_pred)

	# print("\n***** Baseline model *****")
	# print("Training Error = ", trainError)
	# print("Testing Error = ", testError)


	# Development model
	print("\n***** Dev model *****")
	baseline_model_obj = BaselineModel(X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test)
	train_pred, test_pred = dev_model(X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test, baseline_model_obj)
	trainError, testError = calculateError(X_train, train_pred, X_test, test_pred)
	print("\nTraining Error = ", trainError)
	print("Testing Error = ", testError)


if __name__ == "__main__":
	main()