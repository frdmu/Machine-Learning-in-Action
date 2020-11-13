import numpy as np
import matplotlib.pyplot as plt

class perceptron(object):
    def __init__(self, X, y):
        self.X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.y = y
        self.w = np.random.rand(self.X.shape[1])

    # Batch perceptron
    def train(self):
        while True:
            # learning rate 
            alpha = np.random.rand(1)
            # find points that are not be classified correctly
            gradient = 0 
            for i in range(self.X.shape[0]):
                temp = np.dot(self.w, self.X[i]) * self.y[i]
                if temp < 0:
                    gradient += self.X[i] * self.y[i]
            # update self.w
            self.w += alpha * gradient
            # if all the points are classified correctly, finish loop 
            e = np.sum((alpha * gradient) ** 2) 
            print(e) 
            if e < 1e-30:
                break
    
    def plot(self):
        idx_1 = (self.y == 1) 
        x1_1 = self.X[idx_1, 1]
        x2_1 = self.X[idx_1, 2]

        idx_2 = (self.y == -1)
        x1_2 = self.X[idx_2, 1]
        x2_2 = self.X[idx_2, 2]

        plt.axis([-1, 3, -1, 3])

        plt.plot(x1_1, x2_1, 'o')
        plt.plot(x1_2, x2_2, 'x')

        plane_x = np.array([np.min(self.X[:, 1]), np.max(self.X[:, 1])])
        plane_y = (-self.w[0] - self.w[1] * plane_x) / self.w[2]
        plt.plot(plane_x, plane_y)

        plt.show()

if __name__ == '__main__':
    X = [[1, 1],
         [2, 2],
         [2, 0],
         [0, 0],
         [1, 0],
         [0, 1]]
    y = [1, 1, 1, -1, -1, -1]
    X = np.array(X).astype(float)
    y = np.array(y)

    model = perceptron(X, y)
    model.train()
    model.plot()
    print(model.w)
