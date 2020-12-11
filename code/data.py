import numpy as np 
import pandas as pd 


class Data:

	def __init__(self, dataPath):
		self.path = dataPath
		self.trainData = None
		self.testData = None
		self.dataWithNan = None
		self.columnList = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
		self.categoricalColumnList = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex', 'nativeCountry']


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

		xTrain, yTrain = self.trainData.drop("income", axis = 1), self.trainData['income']
		xTest, yTest = self.testData.drop("income", axis = 1), self.testData['income']


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
