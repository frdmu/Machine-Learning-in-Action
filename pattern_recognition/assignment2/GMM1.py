import numpy as np
import matplotlib.pyplot as plt
import csv 
import copy
from scipy.stats import multivariate_normal

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
        i = 0
        while i < self.k:
            self.sigma[i] = np.diag(np.random.rand(dim) + 2)
            i += 1
      
        # probablility P(yi = k | xi, alpha, mu, sigma)
        self.pLabel = np.zeros((self.data.shape[0], k))
    
    def calNormal(self, x):
        tempLabel = np.zeros(self.k)
        for i in range(self.k):
            norm = multivariate_normal(mean=self.mu[i], cov=self.sigma[i]) 
            tempLabel[i] = norm.pdf(x) * self.alpha[i]
        tempLabel /= np.sum(tempLabel)
        return tempLabel

    def train(self):
        e = 1
        # the number of iterate 
        cnt = 0
        while e > 1e-30:
            cnt += 1
            # E step 
            i = 0
            while i < self.data.shape[0]:
                 # calcute P(yi = k | xi, alpha, mu, sigma) 
                self.pLabel[i] = self.calNormal(self.data[i])
                i += 1
            # M step
             # update alpha
            tempAlpha = copy.deepcopy(self.alpha)
            self.alpha = np.mean(self.pLabel, axis=0)

             # update mu
            tempMu = copy.deepcopy(self.mu)
            i = 0
            while i < self.k:
                self.mu[i] = np.array((np.mat(self.pLabel[:, i]) * np.mat(self.data)) / np.sum(self.pLabel[:, i]))
                i += 1

             # update sigma
            tempSigma = copy.deepcopy(self.sigma)
            i = 0
            while i < self.k:
                tempDif = self.data - self.mu[i]
                self.sigma[i] =  np.dot(tempDif.T, tempDif*self.pLabel[:, i].reshape(self.pLabel.shape[0], 1)) / np.sum(self.pLabel[:, i])
                i += 1

            e = np.sum((self.alpha-tempAlpha)**2) + np.sum((self.mu-tempMu)**2) + np.sum((self.sigma-tempSigma)**2)
            print("[{}]:".format(cnt) + "error is {}.".format(e))

    def predict(self, data):
        testData = np.array(data).astype(float)
        predictLabel = np.zeros((testData.shape[0], self.k))
        i = 0
        while i < testData.shape[0]:
            predictLabel[i] = self.calNormal(testData[i])
            i += 1
        return predictLabel
    
    def plot(self, data, label):
        num = data.shape[0]
        idx1 = label[:, 0].reshape(num) > 0.5
        idx2 = label[:, 0].reshape(num) <= 0.5
        
        x1 = data[idx1, 0]
        y1 = data[idx1, 1]
        x2 = data[idx2, 0]
        y2 = data[idx2, 1]

        plt.plot(x1, y1, 'ro', ms=2)
        plt.plot(x2, y2, 'bo', ms=2)
        plt.plot(self.mu[0, 0], self.mu[0, 1], 'bx', ms=15)
        plt.plot(self.mu[1, 0], self.mu[1, 1], 'rx', ms=15)
        plt.show()

def readData(data, file):
    csv_object = csv.reader(open(file, 'r'))
    for row in csv_object:
        data.append(row)

if __name__ == '__main__': 
    trainData = []
    testData = []
    readData(trainData, 'Train2.csv')
    readData(testData, 'Test2.csv')

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


