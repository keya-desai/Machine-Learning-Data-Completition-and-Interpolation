import re
import pandas as pd
import numpy as np

class BaselineModel:
    
    def __init__(self, X_train, X_test, X_prime_train, X_prime_test, isFeatureReal_train, isFeatureReal_test):

        self.X_prime_train = pd.DataFrame(X_prime_train)
        self.X_prime_test = pd.DataFrame(X_prime_test)

        self.isFeatureReal_train = pd.DataFrame(data = isFeatureReal_train)
        self.isFeatureReal_test = pd.DataFrame(data = isFeatureReal_test)

        # Missing value as Nan
        self.X_prime_train = self.X_prime_train.where(self.isFeatureReal_train > 0)
        self.X_prime_test = self.X_prime_test.where(self.isFeatureReal_test > 0)


        self.numFeatures = len(self.isFeatureReal_train.columns)
        self.categoricalFeatures = [[i for i in range(6,13)],
                         [i for i in range(13,29)],
                         [i for i in range(29,36)],
                         [i for i in range(36,50)],
                         [i for i in range(50,56)],
                         [i for i in range(56,61)],
                         [i for i in range(60,63)],
                         [i for i in range(63,104)],
                         [i for i in range(104,106)]]

        self.isCategorical = [i for _l in self.categoricalFeatures for i in _l]
        self.startIndex = [l[0] for l in self.categoricalFeatures]
        self.endIndex = [l[-1] for l in self.categoricalFeatures]

        # Dictionary to store mean/mode of numFeatures
        self.colToMeanMode = {}
        
    def calculateSampleAverage(self):
        
        for col in range(self.numFeatures):
            # Categorical feature 
            if col in self.startIndex:
                start = col
                i = self.startIndex.index(col)
                end = self.endIndex[i]
                df = self.X_prime_train.iloc[:,start:end+1]
                # Determine mode class
                colSum = np.array([df[c].sum() for c in df.columns])
#                 print("colSum",colSum)
                maxClass = np.argmax(colSum)
#                 print("maxClass:", maxClass)
                for i in range(start, end+1):
                    self.colToMeanMode[i] = 0
                self.colToMeanMode[maxClass+start] = 1
                
            elif col not in self.isCategorical:
                # Mean of numerical feature
                mean = self.X_prime_train[col].mean()
                self.colToMeanMode[col] = mean
            else:
                continue
        # print(self.colToMeanMode)
    
    def fillMissingValues(self, colIdxList = None):
        train_pred = self.X_prime_train
        test_pred = self.X_prime_test

        print(train_pred.loc[train_pred[0] == 0])
        
        if colIdxList:
            for col in colIdxList:
                train_pred.fillna(value = self.colToMeanMode[col], inplace = True)
                test_pred.fillna(value = self.colToMeanMode, inplace= True)
        else:
            train_pred.fillna(value = self.colToMeanMode, inplace = True)
            test_pred.fillna(value = self.colToMeanMode, inplace= True)
        return train_pred, test_pred
        
