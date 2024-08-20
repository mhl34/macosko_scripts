import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from tqdm import tqdm
import igraph as igp
import leidenalg as la
import pickle

print("load knn")
embedding0 = np.load('embedding_mat_trial_0.npz')['embeddings']
embedding1 = np.load('embedding_mat_trial_3.npz')['embeddings']
knn_output = np.load('knn_output.npz')
knn_indices = knn_output['knn_indices']
knn_dists = knn_output['knn_dists']

print("getting edges")
edges = []
for i in knn_indices:
    for x, y in zip(i[0] * np.ones_like(i[1:]), i[1:]):
        edges.append((x, y))

print("making graph")
g = igp.Graph(edges=edges, directed=True)

# Apply Leiden algorithm
print("leiden")
# Adjusting Leiden algorithm parameters
resolution_parameter = 8 # Controls the granularity of the clustering
partition_type = la.RBConfigurationVertexPartition  # Type of partitioning method

# Apply Leiden algorithm with custom parameters
partition = la.find_partition(
    g, 
    partition_type,
    resolution_parameter=resolution_parameter,
)

# Print out the cluster assignments
print("save")
with open('partition.pkl', 'wb') as f:
    pickle.dump(partition.membership, f)
