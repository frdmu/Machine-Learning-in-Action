import numpy as np
import matplotlib.pyplot as plt
import copy

class KMeans(object):
    def __init__(self, data):
        self.data = np.array(data).astype(float) 
        self.centroids = self.data[np.random.randint(0, self.data.shape[0], 2), :]
        self.label = np.random.randint(0, 2, self.data.shape[0])
    def EM(self):
        centroids_error = float('inf') 
        data_row = self.data.shape[0]
        centroids_row = self.centroids.shape[0] 
        while centroids_error > 1e-20:
            # E step, clustering
            for i in range(0, data_row): 
                dist = np.sqrt(np.sum((self.data[i, :] - self.centroids)**2, axis=1)) 
                self.label[i] = np.argmin(dist)
            # M step, recalculate random centers after clustering
            pre_centroids = copy.deepcopy(self.centroids)
            for j in range(0, centroids_row): 
                n = self.label[self.label == j].size
                self.centroids[j] = np.sum(self.data[self.label == j, :], axis=0)/n
            centroids_error = np.sum((pre_centroids-self.centroids)**2)
    def plot(self):
        x0 = self.data[self.label==0, 0]
        y0 = self.data[self.label==0, 1]
        x1 = self.data[self.label==1, 0]
        y1 = self.data[self.label==1, 1]

        plt.axis([-1,10,-1,10])
        plt.plot(x0, y0, 'go')
        plt.plot(x1, y1, 'bo')
        plt.plot(self.centroids[0][0], self.centroids[0][1], 'r+', ms=10)
        plt.plot(self.centroids[1][0], self.centroids[1][1], 'r+', ms=10)
        plt.show()

if __name__ == '__main__':
    data = [[0,0],[1,0],[0,1],[1,1],
            [2,1],[1,2],[2,2],[3,2],
            [6,6],[7,6],[8,6],[7,7],
            [8,7],[9,7],[7,8],[8,8],
            [9,8],[8,9],[9,9]]
    test = KMeans(data)
    test.EM()
    test.plot()
