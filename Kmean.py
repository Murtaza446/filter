import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Importing the dataset
df = pd.read_csv('Mall_Customers.csv')

# Extracting Variables
x = df.iloc[:, [3, 4]].values

# Finding the optimal number of clusters using the elbow method
wcss_list = []  # Initializing the list for the values of WCSS
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)

# Plotting the Elbow Method Graph
plt.plot(range(1, 11), wcss_list)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Normalize features for better clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Training the K-means model on the dataset
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(X_scaled)

# Visualizing the clusters
plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s=100, c='blue', label='Cluster 1')  # Cluster 1
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s=100, c='green', label='Cluster 2')  # Cluster 2
plt.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s=100, c='red', label='Cluster 3')  # Cluster 3
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s=100, c='black', label='Cluster 4')  # Cluster 4
plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s=100, c='purple', label='Cluster 5')  # Cluster 5

# Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroid')

# Final plot details
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()