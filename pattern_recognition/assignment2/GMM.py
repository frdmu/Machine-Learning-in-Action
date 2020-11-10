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
            self.sigma[i] = np.diag(np.random.rand(self.dim) + 2)
            i += 1
      
        # probablility P(yi = k | xi, alpha, mu, sigma)
        self.P = np.zeros((self.data.shape[0], self.k))
    
    def calP(self, x):
        tempP = np.zeros(self.k)
        for i in range(self.k):
            norm = multivariate_normal(mean=self.mu[i], cov=self.sigma[i]) 
            tempP[i] = norm.pdf(x) * self.alpha[i]
        tempP /= np.sum(tempP)
        return tempP

    def calPro(self, x):
        tempP = np.zeros(self.k)
        for i in range(self.k):
            norm = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            tempP[i] = norm.pdf(x) * self.alpha[i]
        return np.sum(tempP)

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
                self.P[i] = self.calP(self.data[i])
                i += 1
            # M step
             # update alpha
            tempAlpha = copy.deepcopy(self.alpha)
            self.alpha = np.mean(self.P, axis=0)

             # update mu
            tempMu = copy.deepcopy(self.mu)
            i = 0
            while i < self.k:
                self.mu[i] = np.array((np.mat(self.P[:, i]) * np.mat(self.data)) / np.sum(self.P[:, i]))
                i += 1

             # update sigma
            tempSigma = copy.deepcopy(self.sigma)
            i = 0
            while i < self.k:
                tempDif = self.data - self.mu[i]
                self.sigma[i] =  np.dot(tempDif.T, tempDif*self.P[:, i].reshape(self.P.shape[0], 1)) / np.sum(self.P[:, i])
                i += 1

            e = np.sum((self.alpha-tempAlpha)**2) + np.sum((self.mu-tempMu)**2) + np.sum((self.sigma-tempSigma)**2)
            print("[{}]:".format(cnt) + "error is {}.".format(e))

    def savePara(self, alphaFile, muFile, sigmaFile):
        writer1 = csv.writer(open(alphaFile, 'w'))
        writer1.writerow(self.alpha)
        writer2 = csv.writer(open(muFile, 'w'))
        for row in self.mu:
            writer2.writerow(row)
        writer3 = csv.writer(open(sigmaFile, 'w'))
        for row in self.sigma:
            row = row.reshape(1, row.size)[0]
            writer3.writerow(row)
   
    def loadPara(self, alphaFile, muFile, sigmaFile):
        tempAlpha = []
        tempMu = []
        tempSigma = []
        readData(tempAlpha, alphaFile)
        readData(tempMu, muFile)
        readData(tempSigma, sigmaFile)

        self.alpha = np.array(tempAlpha[0]).astype(float)
        self.mu = np.array(tempMu).astype(float)
        i = 0
        while i < self.k:
            self.sigma[i] = np.array(tempSigma[i]).reshape(self.dim, self.dim)
            i += 1
        self.sigma.astype(float)

def readData(data, file):
    csv_object = csv.reader(open(file, 'r'))
    for row in csv_object:
        data.append(row)

def predict(testData, modelSet, w):
    testData = np.array(testData).astype(float)
    predictPro = np.zeros((testData.shape[0], len(w)))
    i = 0
    while i < testData.shape[0]:
        j = 0
        while j < len(w):
            predictPro[i, j] = modelSet[j].calPro(testData[i]) * w[j]
            j += 1
        i += 1
    return np.argmax(predictPro, axis=1)

def getAccuracy(l, flag):
    idx0 = l==0
    idx1 = l==1
    temp = np.ones(len(l))
    total0 = np.sum(temp[idx0])
    total1 = np.sum(temp[idx1])
    if flag == 0:
        return total0 / (total0 + total1)
    else: 
        return total1 / (total0 + total1)

if __name__ == '__main__': 
#================================
# The simulated data test section
#================================
    trainData1 = []
    testData1 = []
    readData(trainData1, 'Train1.csv')
    readData(testData1, 'Test1.csv')

    trainData2 = []
    testData2 = []
    readData(trainData2, 'Train2.csv')
    readData(testData2, 'Test2.csv')
    
    model1 = GMM(trainData1, 2, 2)
    model2 = GMM(trainData2, 2, 2)
    
    model1.train()
    model2.train()

    model1.savePara('para/alpha1.csv', 'para/mu1.csv', 'para/sigma1.csv')
    model2.savePara('para/alpha2.csv', 'para/mu2.csv', 'para/sigma2.csv')
  
    #model1.loadPara('para/alpha1.csv', 'para/mu1.csv', 'para/sigma1.csv')
    #model2.loadPara('para/alpha2.csv', 'para/mu2.csv', 'para/sigma2.csv')

    modelSet = [model1, model2]
    w = np.array([0.5, 0.5])
    
    print(getAccuracy(predict(testData1, modelSet, w), 0))
    print(getAccuracy(predict(testData2, modelSet, w), 1))



