import numpy as np
import matplotlib.pyplot as plt
import csv 
import copy
from scipy.stats import multivariate_normal

class GMM(object):
    def __init__(self, data, k, dim):
        self.data = np.array(data).astype(float)/100
         # Gaussian number
        self.k = k
         # the number of data feature,ie. data dimension
        self.dim = dim
       
        # initialize parameters: alpha, mu, sigma
        self.alpha = np.random.rand(self.k)
        self.alpha /= np.sum(self.alpha)

        tempMax = np.amax(self.data)
        tempMin = np.amin(self.data)
        self.mu = np.random.uniform(tempMin-1, tempMax+1, self.k*self.dim).reshape(self.k, self.dim)

        self.sigma = np.zeros((self.k, self.dim, self.dim))
        i = 0
        while i < self.k:
            self.sigma[i] = np.diag(np.random.rand(self.dim) + 1000)
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
        while e > 1e-10:
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

def MnistDataDivide(MnistData, MnistLabel):
    MnistData = np.array(MnistData).astype(float) 
    MnistLabel = np.array(MnistLabel).T[0].astype(float)
    data =  []
    i = 0 
    while i < 10:
        idx = MnistLabel == i
        data.append(MnistData[idx])
        i += 1
    return data

def compare(predictMnistLabel, MnistLabel):
    print("type: {}".format(type(MnistLabel))) 
    MnistLabel = np.array(MnistLabel).T[0].astype(float)
    print("MnistLabel.shape: {}".format(MnistLabel.shape)) 
    temp = np.ones(MnistLabel.shape[0]) 
    return np.sum(temp[predictMnistLabel == MnistLabel]) / MnistLabel.shape[0]

if __name__ == '__main__': 
	"""
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
  
    modelSet = [model1, model2]
    w = np.array([0.5, 0.5])
    
    print(getAccuracy(predict(testData1, modelSet, w), 0))
    print(getAccuracy(predict(testData2, modelSet, w), 1))
	"""

#=============================
# MNIST data set test section
#=============================
    trainMnistData = []
    trainMnistLabel = []
    testMnistData = []
    testMnistLabel = []

    readData(trainMnistData, 'TrainSamples.csv')
    readData(trainMnistLabel, 'TrainLabels.csv')
    readData(testMnistData, 'TestSamples.csv')
    readData(testMnistLabel, 'TestLabels.csv')

    trainMnistDataSet = MnistDataDivide(trainMnistData, trainMnistLabel)

	model0 = GMM(trainMnistDataSet[0], 5, 17)
    model0.train()
	model0.savePara('MnistPara/alpha5_0.csv', 'MnistPara/mu5_0.csv', 'MnistPara/sigma5_0.csv')
	
	model1 = GMM(trainMnistDataSet[1], 5, 17)
    model1.train()
	model1.savePara('MnistPara/alpha5_1.csv', 'MnistPara/mu5_1.csv', 'MnistPara/sigma5_1.csv')
	
	model2 = GMM(trainMnistDataSet[2], 5, 17)
    model2.train()
	model2.savePara('MnistPara/alpha5_2.csv', 'MnistPara/mu5_2.csv', 'MnistPara/sigma5_2.csv')
	
	model3 = GMM(trainMnistDataSet[3], 5, 17)
    model3.train()
	model3.savePara('MnistPara/alpha5_3.csv', 'MnistPara/mu5_3.csv', 'MnistPara/sigma5_3.csv')
	
	model4 = GMM(trainMnistDataSet[4], 5, 17)
    model4.train()
	model4.savePara('MnistPara/alpha5_4.csv', 'MnistPara/mu5_4.csv', 'MnistPara/sigma5_4.csv')
	
	model5 = GMM(trainMnistDataSet[5], 5, 17)
    model5.train()
	model5.savePara('MnistPara/alpha5_5.csv', 'MnistPara/mu5_5.csv', 'MnistPara/sigma5_5.csv')
	
	model6 = GMM(trainMnistDataSet[6], 5, 17)
    model6.train()
	model6.savePara('MnistPara/alpha5_6.csv', 'MnistPara/mu5_6.csv', 'MnistPara/sigma5_6.csv')
	
	model7 = GMM(trainMnistDataSet[7], 5, 17)
    model7.train()
	model7.savePara('MnistPara/alpha5_7.csv', 'MnistPara/mu5_7.csv', 'MnistPara/sigma5_7.csv')
	
	model8 = GMM(trainMnistDataSet[8], 5, 17)
    model8.train()
	model8.savePara('MnistPara/alpha5_8.csv', 'MnistPara/mu5_8.csv', 'MnistPara/sigma5_8.csv')
	
	model9 = GMM(trainMnistDataSet[9], 5, 17)
    model9.train()
	model9.savePara('MnistPara/alpha5_9.csv', 'MnistPara/mu5_9.csv', 'MnistPara/sigma5_9.csv')
    
    #model0.loadPara('MnistPara/alpha5_0.csv', 'MnistPara/mu5_0.csv', 'MnistPara/sigma5_0.csv')
    #model1.loadPara('MnistPara/alpha5_1.csv', 'MnistPara/mu5_1.csv', 'MnistPara/sigma5_1.csv')
    #model2.loadPara('MnistPara/alpha5_2.csv', 'MnistPara/mu5_2.csv', 'MnistPara/sigma5_2.csv')
    #model3.loadPara('MnistPara/alpha5_3.csv', 'MnistPara/mu5_3.csv', 'MnistPara/sigma5_3.csv')
    #model4.loadPara('MnistPara/alpha5_4.csv', 'MnistPara/mu5_4.csv', 'MnistPara/sigma5_4.csv')
    #model5.loadPara('MnistPara/alpha5_5.csv', 'MnistPara/mu5_5.csv', 'MnistPara/sigma5_5.csv')
    #model6.loadPara('MnistPara/alpha5_6.csv', 'MnistPara/mu5_6.csv', 'MnistPara/sigma5_6.csv')
    #model7.loadPara('MnistPara/alpha5_7.csv', 'MnistPara/mu5_7.csv', 'MnistPara/sigma5_7.csv')
    #model8.loadPara('MnistPara/alpha5_8.csv', 'MnistPara/mu5_8.csv', 'MnistPara/sigma5_8.csv')
    #model9.loadPara('MnistPara/alpha5_9.csv', 'MnistPara/mu5_9.csv', 'MnistPara/sigma5_9.csv')

    modelSet = [model0, model1, model2, model3, model4, model5, model6, model7, model8, model9]
    w = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    predictLabel = predict(testMnistData, modelSet, w)
    print(compare(predictLabel, testMnistLabel))
