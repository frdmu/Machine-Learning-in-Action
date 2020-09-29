import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
  dataMat = []
  fr = open(fileName)
  for line in fr.readlines():
    curLine = line.strip().split('\t')
    fltLine = list(map(float, curLine))
    print("fltLine: {}".format(fltLine))
    dataMat.append(fltLine)
  print(dataMat)
  return dataMat

def distEclud(vecA, vecB):
  return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
  n = np.shape(dataSet)[1]
  print(np.shape(dataSet), n)
  centroids = np.mat(np.zeros((k, n)))
  for j in range(n):
    minJ = np.min(dataSet[:, j])
    maxJ = np.max(dataSet[:, j])
    rangeJ = float(maxJ - minJ)
    centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
  #get the numbers of dataset
  m = np.shape(dataSet)[0]
  clusterAssment = np.mat(np.zeros((m, 2)))
  centroids = createCent(dataSet, k)
  clusterChanged = True
  while clusterChanged:
    clusterChanged = False
    for i in range(m):
      minDist = float('inf')
      minIndex = -1
      for j in range(k):
        distJI = distMeas(centroids[j, :], dataSet[i, :])
        if distJI < minDist:
          minDist = distJI
          minIndex = j
      if clusterAssment[i, 0] != minIndex:
        clusterChanged = True
      clusterAssment[i, :] = minIndex, minDist**2
    for cent in range(k):
    # filter out all samples in the dataSet that belong to the current centroid class 
    	ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
    	centroids[cent, :] = np.mean(ptsInClust, axis=0)
  return centroids, clusterAssment

def plotDataSet(filename):
  ## 1.import data
  datMat = np.mat(loadDataSet(filename))
  # print("datMat.shape: {}".format(np.shape(datMat)))

  ## 2.k means clustering, let k = 4
  myCentroids, clustAssing = kMeans(datMat, 4)
  
  datMat = datMat.tolist()
  clustAssing = clustAssing.tolist()
  myCentroids = myCentroids.tolist()
  #print('datMat:{}'.format(datMat))
  #print('clustAssing:{}'.format(clustAssing))
  #print('myCentroids:{}'.format(myCentroids))
  
  Xcord = [[], [], [], []]
  Ycord = [[], [], [], []]
  
  m = len(clustAssing)
  for i in range(m):
    if int(clustAssing[i][0]) == 0:
      Xcord[0].append(datMat[i][0])
      print('datMat[{}][]:{}'.format(i, datMat[i][0]))
      Ycord[0].append(datMat[i][1])
    elif int(clustAssing[i][0]) == 1:
      Xcord[1].append(datMat[i][0])
      print('datMat[{}][]:{}'.format(i, datMat[i][0]))
      Ycord[1].append(datMat[i][1])
    elif int(clustAssing[i][0]) == 2:
      Xcord[2].append(datMat[i][0])
      print('datMat[{}][]:{}'.format(i, datMat[i][0]))
      Ycord[2].append(datMat[i][1])
    elif int(clustAssing[i][0]) == 3:
      Xcord[3].append(datMat[i][0])
      print('datMat[{}][]:{}'.format(i, datMat[i][0]))
      Ycord[3].append(datMat[i][1])
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(Xcord[0], Ycord[0], s=20, c='b', marker='*', alpha=.5)
  ax.scatter(Xcord[1], Ycord[1], s=20, c='r', marker='D', alpha=.5)
  ax.scatter(Xcord[2], Ycord[2], s=20, c='c', marker='>', alpha=.5)
  ax.scatter(Xcord[3], Ycord[3], s=20, c='k', marker='o', alpha=.5)

  ax.scatter(myCentroids[0][0], myCentroids[0][1], s=100, c='k', marker='+', alpha=.5)
  ax.scatter(myCentroids[1][0], myCentroids[1][1], s=100, c='k', marker='+', alpha=.5)
  ax.scatter(myCentroids[2][0], myCentroids[2][1], s=100, c='k', marker='+', alpha=.5)
  ax.scatter(myCentroids[3][0], myCentroids[3][1], s=100, c='k', marker='+', alpha=.5)
  plt.title('DataSet')
  plt.xlabel('X')
  plt.show()

if __name__ == '__main__':
  plotDataSet('./testSet.txt')
