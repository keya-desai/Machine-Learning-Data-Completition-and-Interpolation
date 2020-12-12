import numpy as np 
import pandas as pd 


class Data:

	def __init__(self, dataPath):
		self.path = dataPath
		self.trainData = None
		self.testData = None
		self.dataWithNan = None
		self.columnList = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
		self.categoricalColumnList = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry', 'income']


	def getData(self):

		self.trainData = pd.read_csv(self.path + 'adult.data', header = None)
		self.trainData.columns = self.columnList

		self.testData = pd.read_csv(self.path + 'adult.test', skiprows = 1, header = None)
		self.testData.columns = self.columnList

		# Preprocessing Data
		# Removing NaN values 
		self.removeNan(self.trainData)
		self.removeNan(self.testData)

		print("Train Data shape = ", self.trainData.shape)
		print("Test Data shape = ", self.testData.shape)
		print("Data with Nan shape = ", self.dataWithNan.shape)

		# One hot encoding
		self.trainData = self.oneHotEncoding(self.trainData)
		# print(self.trainData.head())
		# (30162, 105)
		print(self.trainData.shape)

		self.testData = self.oneHotEncoding(self.testData)
		# print(self.testData.head())
		# (15060, 104)
		print(self.testData.shape)

		# 1 column in missing in test data - add feature manually
		for column in self.trainData.columns:
			if column not in self.testData.columns:
				print("Adding dummy column {} in Test data".format(column))
				self.testData[column] = 0
		print(self.testData.shape)

		X = self.trainData
		X_prime, isFeatureReal = self.prepareMissingData(X)
		print(isFeatureReal)
		print(isFeatureReal.shape)
		print("Total missing = ", np.sum(isFeatureReal==0))

		print(X_prime)
		print(X_prime.shape)

		# xTrain, yTrain = self.trainData.drop("income", axis = 1), self.trainData['income']
		# xTest, yTest = self.testData.drop("income", axis = 1), self.testData['income']


	def prepareMissingData(self, df, nMissingMax = 3):

		m, k = df.shape
		isFeatureReal = pd.DataFrame(1, columns = df.columns, index = df.index)

		continuousColList = [x for x in self.columnList if x not in self.categoricalColumnList]

		# Iterating each row
		for idx in range(m):

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
data.getData()
