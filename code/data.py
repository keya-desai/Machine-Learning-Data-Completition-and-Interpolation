import numpy as np 
import pandas as pd 
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class Data:

	def __init__(self, dataPath):
		self.path = dataPath
		self.data = None
		# self.testData = None
		self.dataWithNan = None
		self.columnList = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
		self.categoricalColumnList = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry', 'income']


	def getData(self, newData = False):

		if newData:

			self.data = pd.read_csv(self.path + 'adult.data', header = None)
			self.data.columns = self.columnList

			# self.testData = pd.read_csv(self.path + 'adult.test', skiprows = 1, header = None)
			# self.testData.columns = self.columnList

			# Preprocessing Data
			# Removing NaN values 
			self.removeNan(self.data)
			# self.removeNan(self.testData)

			# One hot encoding
			self.data = self.oneHotEncoding(self.data)

			print("Data shape = ", self.data.shape)
			# print("Test Data shape = ", self.testData.shape)
			print("Data with Nan shape = ", self.dataWithNan.shape)

			
			# print(self.data.head())
			# (30162, 105)
			# self.data = self.data.head(1000)
			print(self.data.shape)

			# self.testData = self.oneHotEncoding(self.testData)
			# print(self.testData.head())
			# (15060, 104)
			# print(self.testData.shape)

			# 1 column in missing in test data - add feature manually
			# for column in self.data.columns:
				# if column not in self.testData.columns:
					# print("Adding dummy column {} in Test data".format(column))
					# self.testData[column] = 0
			# print(self.testData.shape)


			X_train, X_test = train_test_split(self.data, test_size = 0.20, random_state = 42)
			print(X_train.shape, X_test.shape)
			# exit()
			X_prime_train, isFeatureReal_train = self.prepareMissingData(X_train)
			X_prime_test, isFeatureReal_test = self.prepareMissingData(X_test)

			# Conver X_train and X_test to numpy
			X_train = X_train.to_numpy()
			X_test = X_test.to_numpy()

			# X_prime, isFeatureReal = self.prepareMissingData(self.data)
			# X_train, X_test, X_missing_train, X_missing_test = train_test_split(X, X_prime, test_size=0.20, random_state=42)



			# with open(self.path + 'x.npy', 'wb') as f:
			np.save(self.path + 'Train/X.npy', X_train)
			np.save(self.path + 'Test/X.npy', X_test)
			# with open(self.path + 'x_prime.npy', 'wb') as f:
			np.save(self.path + 'Train/X_prime.npy', X_prime_train)
			np.save(self.path + 'Test/X_prime.npy', X_prime_test)
			# with open(self.path + 'feature_information.npy', 'wb') as f:
			np.save(self.path + 'Train/feature_information.npy', isFeatureReal_train)
			np.save(self.path + 'Test/feature_information.npy', isFeatureReal_test)

		else:
			# X = np.load(self.path + 'x_prime.npy', allow_pickle = True)	
			# X_prime = np.load(self.path + 'x_prime.npy', allow_pickle = True)
			# isFeatureReal = np.load(self.path + 'feature_information.npy', allow_pickle = True)

			X_train = np.load(self.path + 'Train/X.npy', allow_pickle = True)
			X_test = np.load(self.path + 'Test/X.npy', allow_pickle = True)

			X_prime_train = np.load(self.path + 'Train/X_prime.npy', allow_pickle = True)
			X_prime_test = np.load(self.path + 'Test/X_prime.npy', allow_pickle = True)

			isFeatureReal_train = np.load(self.path + 'Train/feature_information.npy', allow_pickle = True)
			isFeatureReal_test = np.load(self.path + 'Test/feature_information.npy', allow_pickle = True)


			# print(isFeatureReal)
			# print(isFeatureReal.shape)
			# print("Total missing = ", np.sum(isFeatureReal==0))

			# print(X_prime)
			# print(X_prime.shape)

		return X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test


	def prepareMissingData(self, df, nMissingMax = 3):
		pd.options.mode.chained_assignment = None

		m, k = df.shape
		isFeatureReal = pd.DataFrame(1, columns = df.columns, index = df.index)

		continuousColList = [x for x in self.columnList if x not in self.categoricalColumnList]

		# Iterating each row
		for idx in tqdm(range(m)):

			row = df.iloc[idx]

			# Filtering out the categorical columns which are zero
			features = row[row!=0]
			# Adding back the continuous columns whose value = 0
			for column in continuousColList:
			    if column not in features.keys():
			        features[column] = row[column]

			# Randomly selecting number of columns missing in this row out of [0, nMissingMax]
			nMissing = np.random.randint(0, nMissingMax + 1)
			# Randomly selecting features which will be missing - return list
			x = np.random.choice(features.keys(), nMissing, replace = False)


			for col in x:
				# If column is continuous then can directly set it to 0. 
				if col in continuousColList:
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


	def oneHotEncoding(self, df):

		for column in self.categoricalColumnList:
			y = pd.get_dummies(df[column], prefix='Is_' + column)
			df = pd.concat([df, y], axis = 1)
			df.drop(column, axis = 1, inplace = True)
		return df

	def removeNan(self, df):

		self.dataWithNan = pd.DataFrame(columns = df.columns)
    
		for col in df.columns:
		    subset = df[df[col] == " ?"]
		    self.dataWithNan = self.dataWithNan.append(subset)
		    df.drop(subset.index, inplace = True)


data = Data('../Data/')
X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test = data.getData(newData = False)
print("X train shape = ", X_train.shape)
print("X test shape = ", X_test.shape)
print("X prime train shape = ", X_prime_train.shape)
print("X prime test shape = ", X_prime_test.shape)
print("feature information train shape = ", isFeatureReal_train.shape)
print("feature information test shape = ", isFeatureReal_test.shape)