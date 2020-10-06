import matplotlib.pyplot as plt
import numpy as np
import KMeans
import math 

"""
Function description: spherical distance	
"""
def distSLC(vecA, vecB):
	a = math.sin(vecA[0, 1] * np.pi / 180) * math.sin(vecB[0, 1] * np.pi / 180)
	b = math.cos(vecA[0, 1] * np.pi / 180) * math.cos(vecB[0, 1] * np.pi / 180) * math.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
	return math.acos(a + b) * 6371.0

 
def clusterClubs(numClust=5):
	datList = []
	for line in open('places.txt').readlines():
		lineArr = line.split('\t')
		datList.append([float(lineArr[4]), float(lineArr[3])])
	datMat = np.mat(datList)

	myCentroids, clustAssing = KMeans.biKmeans(datMat, numClust, distMeas=distSLC)
	
	fig = plt.figure()
	rect = [0.1, 0.1, 0.8, 0.8]
	scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']	
	axprops = dict(xticks=[], yticks=[])
	ax0 = fig.add_axes(rect, label='ax0', **axprops)
	imgP = plt.imread('Portland.png')
	ax0.imshow(imgP)
	ax1 = fig.add_axes(rect, label='ax1', frameon=False)
	for i in range(numClust):
		ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
		markerStyle = scatterMarkers[i % len(scatterMarkers)]
		ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], \
					ptsInCurrCluster[:, 1].flatten().A[0], \
					marker = markerStyle, s=90)
	for i in range(numClust):
		ax1.scatter(myCentroids[i].tolist()[0][0], myCentroids[i].tolist()[0][1], s=300, c='k', marker='+', alpha=.5)
	plt.show()


if __name__ == '__main__':
	clusterClubs()
