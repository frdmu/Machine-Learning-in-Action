import numpy as np
import matplotlib.pyplot as plt
import csv 
import copy

class GMM(object):
    def __init__(self, data, k, dim):
        self.data = np.array(data).astype(float)
         # the number of clustering 
        self.k = k
         # the number of data feature,ie. data dimension
        self.dim = dim
       
        # initialize parameters: alpha, mu, sigma
        self.alpha = np.random.rand(k)
        self.alpha /= np.sum(self.alpha)

        tempMax = np.amax(self.data)
        tempMin = np.amin(self.data)
        self.mu = np.random.uniform(tempMin-1, tempMax+1, self.k*self.dim).reshape(self.k, self.dim)

        self.sigma = np.zeros((self.k, self.dim, self.dim))
         # self.sigma's inverse matrix 
        self.sigmaI = np.zeros((self.k, self.dim, self.dim))
        i = 0
        while i < self.k:
            self.sigma[i] = np.diag(np.random.rand(dim) + 2)
            i += 1
        i = 0
        while i < self.k:
            self.sigmaI[i] = np.array(np.mat(self.sigma[i]).I)
             i += 1
       
       # Normalized 
       self.pLabel = np.random.rand(self.data.shape[0], k)
       tempSum = np.sum(self.pLabel, axis=1)
       self.pLabel = np.divide(self.pLabel.transpose(), tempSum).transpose()



def readData(data, file):
    csv_object = csv.reader(open(file, 'r'))
    for row in csv_object:
        data.append(row)

if __name__ == '__main__': 
    trainData = []
    testData = []
    readData(trainData, 'Train1.csv')
    readData(testData, 'Test1.csv')

    model = GMM(trainData, 2, 2)
    model.train()
    print("alpha:") 
    print(model.alpha)
    print("mu:") 
    print(model.mu)
    print("sigma:") 
    print(model.sigma)

    testData = np.array(testData).astype(float)
    label = model.predict(testData)
    model.plot(testData, label)


