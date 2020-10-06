from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if __name__ == '__main__':
	x, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)
	plt.scatter(x[:, 0], x[:, 1], marker='o', color='blue')
	
	km = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=0)
	y_km = km.fit_predict(x)
	
	plt.scatter(x[y_km==0, 0], x[y_km==0, 1], s=50, c='orange', marker='o', label='cluster 1')	
	plt.scatter(x[y_km==1, 0], x[y_km==1, 1], s=50, c='green', marker='s', label='cluster 2')
	plt.scatter(x[y_km==2, 0], x[y_km==2, 1], s=50, c='blue', marker='^', label='cluster 3')

	plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker="*", c="red", label="cluster center")
	plt.legend()
	plt.grid()
	plt.show()
