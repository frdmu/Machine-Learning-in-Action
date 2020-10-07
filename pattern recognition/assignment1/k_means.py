import matplotlib.pyplot as plt
import numpy as np
#计算距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))
#得到k个随机质心
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))

    for j in range(n):
        minJ = np.min(dataSet[:, j])    
        maxJ = np.max(dataSet[:, j])    
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids
"""
函数：k-均值算法
入口参数：dataSet:   参与聚类的数据集
		  k:		 聚类个数
		  disMeas:	 计算数据集中的点与质心之间的距离
		  createCent:得到k个随机质心
返回参数：centroids:  聚类完成以后的质心
		  clusterAssment: 聚类结果，包括 1.某点所属类别 2.与质心的距离的平方(目前好像没用)
"""
def KMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
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
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)    
    return centroids, clusterAssment    
#画出聚类结果
def plotDataSet(dataSet):
    Centroids, clusterAssment = KMeans(dataSet, 2)

    dataSet = dataSet.tolist()
    Centroids = Centroids.tolist()
    clusterAssment = clusterAssment.tolist()

    xcord = [[], []]
    ycord = [[], []]

    m = len(clusterAssment)
    for i in range(m):
        if int(clusterAssment[i][0]) == 0:
            xcord[0].append(dataSet[i][0])
            ycord[0].append(dataSet[i][1])
        elif int(clusterAssment[i][0]) == 1:    
            xcord[1].append(dataSet[i][0])
            ycord[1].append(dataSet[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord[0], ycord[0], s=20, c='b', marker='*', alpha=.5)
    ax.scatter(xcord[1], ycord[1], s=20, c='r', marker='o', alpha=.5)
    ax.scatter(Centroids[0][0], Centroids[0][1], s=100, c='k', marker='+', alpha=.5)
    ax.scatter(Centroids[1][0], Centroids[1][1], s=100, c='k', marker='+', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    data = [[0, 0], [1, 0], [0, 1], [1, 1],
            [2, 1], [1, 2], [2, 2], [3, 2],
            [6, 6], [7, 6], [8, 6], [7, 7],
            [8, 7], [9, 7], [7, 8], [8, 8],
            [9, 8], [8, 9], [9, 9]]
    dataSet = np.mat(data)
    plotDataSet(dataSet)    
    
