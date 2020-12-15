import numpy as np 
import pandas as pd 
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Data:

	def __init__(self, dataPath):
		self.path = dataPath
		self.trainData = None
		self.testData = None
		self.dataWithNan = None
		self.columnList = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
		self.categoricalColumnList = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry', 'income']
		self.continuousColumnList = [x for x in self.columnList if x not in self.categoricalColumnList]

	def getData(self, newData = False):

		if newData:

			print("Reading data")
			self.trainData = pd.read_csv(self.path + 'adult.data', header = None)
			self.trainData.columns = self.columnList

			self.testData = pd.read_csv(self.path + 'adult.test', skiprows = 1, header = None)
			self.testData.columns = self.columnList

			# Preprocessing trainData
			print("Preprocessing data")
			# Removing NaN values 
			self.removeNan(self.trainData)
			self.removeNan(self.testData)

			# normalize continuous features
			self.normalizeFeatures(self.trainData)
			self.normalizeFeatures(self.testData)

			# One hot encoding
			self.trainData = self.oneHotEncoding(self.trainData)
			self.testData = self.oneHotEncoding(self.testData)

			# Reordering columns of testdata to match traindata
			self.testData.reindex(columns = self.trainData.columns)

			# print("trainData shape = ", self.trainData.shape)
			# print("Test trainData shape = ", self.testData.shape)
			# print("trainData with Nan shape = ", self.dataWithNan.shape)

			# self.trainData = self.trainData.head(1000)
			# self.testData = self.testData.head(1000)
			# print(self.trainData.shape)
			# print(self.testData.shape)

			# 1 column in missing in test trainData - add feature manually
			for column in self.trainData.columns:
				if column not in self.testData.columns:
					print("Adding dummy column {} in Test Data".format(column))
					self.testData[column] = 0

			print("\nRandomly removing features from training data")
			X_prime_train, isFeatureReal_train = self.prepareMissingData(self.trainData)
			print("\nRandomly removing features from testing data")
			X_prime_test, isFeatureReal_test = self.prepareMissingData(self.testData)

			# Conver X_train and X_test to numpy
			X_train = self.trainData.to_numpy()
			X_test = self.testData.to_numpy()

			print("\nWriting training data in the folder {}".format(self.path + 'Train/'))
			print("\nWriting testing data in the folder {}".format(self.path + 'Test/'))
			np.save(self.path + 'Train/X.npy', X_train)
			np.save(self.path + 'Test/X.npy', X_test)

			np.save(self.path + 'Train/X_prime.npy', X_prime_train)
			np.save(self.path + 'Test/X_prime.npy', X_prime_test)

			np.save(self.path + 'Train/feature_information.npy', isFeatureReal_train)
			np.save(self.path + 'Test/feature_information.npy', isFeatureReal_test)

		else:

			print("Reading training data from the folder {}".format(self.path + 'Train/'))
			print("Reading testing data from the folder {}".format(self.path + 'Test/'))

			X_train = np.load(self.path + 'Train/X.npy', allow_pickle = True)
			X_test = np.load(self.path + 'Test/X.npy', allow_pickle = True)

			X_prime_train = np.load(self.path + 'Train/X_prime.npy', allow_pickle = True)
			X_prime_test = np.load(self.path + 'Test/X_prime.npy', allow_pickle = True)

			isFeatureReal_train = np.load(self.path + 'Train/feature_information.npy', allow_pickle = True)
			isFeatureReal_test = np.load(self.path + 'Test/feature_information.npy', allow_pickle = True)

		return X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test


	def prepareMissingData(self, df, nMissingMax = 3):
		
		# Supress the warning
		pd.options.mode.chained_assignment = None
		m, k = df.shape
		isFeatureReal = pd.DataFrame(1, columns = df.columns, index = df.index)
		

		# Iterating each row
		for idx in tqdm(range(m)):

			row = df.iloc[idx]

			# Filtering out the categorical columns which are zero
			features = row[row!=0]
			# Adding back the continuous columns whose value = 0
			for column in self.continuousColumnList:
			    if column not in features.keys():
			        features[column] = row[column]

			# Randomly selecting number of columns missing in this row out of [0, nMissingMax]
			nMissing = np.random.randint(0, nMissingMax + 1)
			# Randomly selecting features which will be missing - return list
			x = np.random.choice(features.keys(), nMissing, replace = False)


			for col in x:
				# If column is continuous then can directly set it to 0. 
				if col in self.continuousColumnList:
				    isFeatureReal[col].iloc[idx] = 0
				    row[col] = 0
				# For categorical column, need to set 0 in isFeatureReal in all columns 
				# For example - if col == "Is_sex_male", we will set 0 in both 
				# "Is_sex_Male" and "Is_sex_Female" in the df isFeatureMissing. 
				else:		
					row[col] = 0
					# Gives ["Is", "sex", "female"] -> "sex"
					substr = col.split('_')[1]
					# colsCategory = ["Is_sex_male", "Is_sex_female"]
					colsCategory = row.filter(like= substr)
					# setting both columns values to 0. 
					for c in colsCategory.keys():
					    isFeatureReal[c].iloc[idx] = 0
	                
			df.iloc[idx] = row

		df = df.to_numpy()
		isFeatureReal = isFeatureReal.to_numpy()

		return df, isFeatureReal

	def removeNan(self, df):

		self.dataWithNan = pd.DataFrame(columns = df.columns)
    
		for col in df.columns:
		    subset = df[df[col] == " ?"]
		    self.dataWithNan = self.dataWithNan.append(subset)
		    df.drop(subset.index, inplace = True)


	def oneHotEncoding(self, df):

		for column in self.categoricalColumnList:
			y = pd.get_dummies(df[column], prefix='Is_' + column)
			df = pd.concat([df, y], axis = 1)
			df.drop(column, axis = 1, inplace = True)
		return df

	def normalizeFeatures(self, df):

		for col in self.continuousColumnList:
			maxValue = max(df[col])
			minValue = min(df[col])
			df[col] = (df[col] - minValue)/(maxValue - minValue)

		return df



