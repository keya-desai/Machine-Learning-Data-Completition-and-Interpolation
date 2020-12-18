import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt

class knnModel:
    
    def __init__(self, trainData, isFeatureMissingTrain, testData, isFeatureMissingTest, categoricalFeatures, n = 5):
        self.features = isFeatureMissingTrain.columns
        self.trainData = trainData
        self.testData = testData
        self.num_neighbors = n
        self.categoricalFeatures = categoricalFeatures
        self.isCategorical = [i for _l in self.categoricalFeatures for i in _l]
        self.startIndex = [l[0] for l in self.categoricalFeatures]
        self.endIndex = [l[-1] for l in self.categoricalFeatures]
        
        self.trainData = self.trainData.where(isFeatureMissingTrain > 0)
        self.testData = self.testData.where(isFeatureMissingTest > 0)
    
#     def calculateSimilarity(self, x1, x2):
#         # Not keeping categorical (Hamming) and Euclidean separate
#         simScore = 0
#         for col,val in x1.iteritems():
#             if np.isnan(x1[col]) or np.isnan(x2[col]):
#                 continue
#             else:
#                 simScore += (x1[col] - x2[col])**2
#         return np.sqrt(simScore)

#     def getNeighbors(self, x):
#         sim = [];
#         for index, row in self.trainData.iterrows():
#             simScore = self.calculateSimilarity(row, x)
#             sim.append((index, simScore))
#         sim = sorted(sim, key = lambda x: x[1], reverse=True)
#         return sim
    
    # Using Sklearn similarity for computation speed
    def calculateSimilarity(self, x1, x2):
#         sim = pairwise_distances.cosine_similarity(x1.fillna(0), x2.fillna(0))
        sim = pairwise.nan_euclidean_distances(x1, x2)
        return sim
    
    def predictFeature(self, x, neighbors, n = 5):
        num_neighbors = neighbors.shape[0]
        for k in self.features:
            if np.isnan(x[k]):
                if k in self.startIndex:
                    start = k
                    i = self.startIndex.index(k)
                    end = self.endIndex[i]
                    classes = [0 for i in range(start,end+1)]
                    cnt = 0 
                    for _n in range(num_neighbors):
                        neigh = neighbors[_n]
                        tmp_df = self.trainData.iloc[neigh,start:end+1]
                        if not tmp_df.isnull().any():
                            cnt += 1
                            idx = np.argmax(tmp_df.to_numpy())
                            classes[idx] += 1
                        if cnt == n:
                            break
#                     print("Classes:", classes)
                    maxClass = np.argmax(np.array(classes))
#                     print("maxClass:", maxClass)
                    for i in range(start,end+1):
                        x[i] = 0.0
                    x[maxClass+start] = 1.0
                elif k not in self.isCategorical:
                    sumVal = 0
                    cnt = 0
                    for _n in range(num_neighbors):
                        neigh = neighbors[_n]
                        tmp_df = self.trainData.iloc[neigh,k]
                        if not np.isnan(tmp_df):
                            cnt += 1
                            sumVal += tmp_df
                        if cnt == n:
                            break
                    x[k] = sumVal/n
                else:
                    continue
            else:
                continue
        return x

    def predictData(self):
        i = 0
        # sim = np.load('Data/simScores.npy')
        sim = self.calculateSimilarity(self.testData, self.trainData)
        # np.save('Data/simScores.npy', sim)
        simScores = np.flip(np.argsort(sim, axis = 1), axis =1)
        simScores = np.argsort(sim, axis = 1)
        for idx, row in self.testData.iterrows():
            neighbors = simScores[idx]
            self.testData.iloc[idx] = self.predictFeature(row, neighbors, self.num_neighbors)
            i += 1
        return 
    
    # Root Mean square
    def calculateTestError(self, testTrueData):
        
        testPredicted = self.testData.to_numpy()

        diffTest = (testTrueData - testPredicted)**2
        sumTest = np.sum(diffTest, axis = 1)
        errTest = np.sqrt(np.mean(sumTest))

        return errTest   