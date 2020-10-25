import matplotlib.pyplot as plt
from gmm import *

# set debug mode
DEBUG = True

# load data
Y = np.loadtxt("gmm.data")
matY = np.matrix(Y, copy=True)

# the number of clustering
K = 2

# training the GMM model, return the parameter of GMM model
mu, cov, alpha = GMM_EM(matY, K, 100)

# clustering
N = Y.shape[0]
gamma = getExpectation(matY, mu, cov, alpha)
category = gamma.argmax(axis=1).flatten().tolist()[0]

class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
class2 = np.array([Y[i] for i in range(N) if category[i] == 1])

# plot result
plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
plt.legend(loc="best") 
plt.title("GMM Clustering By EM Algorithm")
plt.show()
