import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv 
import copy

def readData(data, filename):
    csvObject = csv.reader(open(filename, 'r'))
    for row in csvObject:
        data.append(row)

class KMeans(object):
    def __init__(self, data, k):
        self.data = np.array(data).astype(float)
        self.centroids = self.data[np.random.randint(0, self.data.shape[0], k), :]
        self.label = np.zeros((1, self.data.shape[0]))[0]
    def EM(self):
        error = float('inf') 
        dataRow = self.data.shape[0]
        centroidsRow = self.centroids.shape[0] 
        while error > 1e-20: 
            # clustering 
            for i in range(0, dataRow):
                dist = np.sqrt(np.sum((self.data[i, :]-self.centroids)**2, axis=1))
                self.label[i] = np.argmin(dist)
            # recalculate centroids after clustering and calculate the error between precentroids and newcentroids 
            preCentroids = copy.deepcopy(self.centroids)
            for j in range(0, centroidsRow):
                n = self.label[self.label==j].size
                self.centroids[j] = np.sum(self.data[self.label==j, :], axis=0)/n
            error = np.sum((preCentroids-self.centroids)**2)
if __name__ == '__main__':
    data = []
    label = []
    readData(data, 'ClusterSamples.csv')
    readData(label, 'SampleLabels.csv')
    label = np.array(label).astype(int).transpose()
    label = label[0, :]

    test = KMeans(data, 10)
    test.EM()

    labelFrame = np.zeros((10, 10)).astype(int)
    for i in range(0, 10):
        for j in range(0, 10):
            realClusterIndex = np.arange(label.size)[label==i]
            testCluster = test.label[realClusterIndex]
            labelFrame[i, j] = testCluster[testCluster==j].size 
    df = pd.DataFrame(labelFrame, index=range(10), columns=range(10))
    print(df)
