from data import Data
from models.baseline import BaselineModel



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


	# Baseline model
	baseline_model = BaselineModel(X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test)
	baseline_model.calculateSampleAverage()
	baseline_model.fillMissingValues()
	trainError, testError = baseline_model.calculateError(X_train, X_test)

	print("\n***** Baseline model *****")
	print("Training Error = ", trainError)
	print("Testing Error = ", testError)

if __name__ == "__main__":
	main()