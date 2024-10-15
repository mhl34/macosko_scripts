import numpy as np
import matplotlib.pyplot as plt
# from helpers import *
from tqdm import tqdm
import igraph as igp
import leidenalg as la
import pickle
import time
from helpers import *
import scipy
import pandas as pd
import os
from scipy.sparse import coo_matrix
import argparse

parser = argparse.ArgumentParser(description='Parser for KNN')

parser.add_argument('-i', '--in_dir', default='.', dest='in_dir')      # option that takes a value
parser.add_argument('-o', '--out_dir', default='.', dest='out_dir')
parser.add_argument('-n', '--n_neighbors', default=45, dest='n_neighbors'

args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
n_neighbors = int(args.n_neighbors)

print(f'KNN output')
knn_output = np.load(f'{in_dir}/knn.npz')
knn_indices = knn_output['indices']
knn_dists = knn_output['dists']
knn_indices = knn_indices[:, :n_neighbors]
knn_dists = knn_dists[:, :n_neighbors]

# # Check the shape and dimensions
print("KNN indices shape:", knn_indices.shape)
print("KNN distances shape:", knn_dists.shape)

# # Create graph edges with distances
print("Getting edges and weights")
start = time.time()
edges = []
# weights = []

for i, neighbors in enumerate(knn_indices):
    for j, neighbor in enumerate(neighbors[1:]):  # Skip self (first element in neighbors)
        edges.append((i, neighbor))
        # weights.append(knn_dists[i][j + 1])  # Corresponding distance (offset by +1)

# Create the graph and add edge weights
print("Making graph")
g = igp.Graph(edges=edges, directed=True)
# g.es['weight'] = weights  # Add weights to the edges
end = time.time()
print(f'Graph creation: {end - start} seconds')

# Apply Leiden algorithm
print("Running Leiden algorithm with weighted edges")
start = time.time()
resolution_parameter = 160  # Controls the granularity of the clustering
partition_type = la.RBConfigurationVertexPartition  # Type of partitioning method

# Apply Leiden algorithm with custom parameters and edge weights
partition = la.find_partition(
    g,
    partition_type,
    resolution_parameter=resolution_parameter,
    # weights=g.es['weight']  # Pass the edge weights to the partitioning method
)

end = time.time()

np.savez(f'{out_dir}/membership_all_counts.npz', membership = np.array(partition.membership))

print("Leiden partition complete")

print(f'number of clusters: {len(np.unique(partition.membership))}')
print(f'modularity: {partition.modularity}')
print(f'Leiden clustering: {end - start} seconds')

def inter_cluster_edge_calc(partition, g, weighted = True):
    num_clusters = len(np.unique(partition.membership))
    inter_cluster_edges = np.zeros((num_clusters, num_clusters))

    # Count the number of edges between clusters
    mem = np.array(partition.membership)
    for i in tqdm(range(len(g.es))):
        edge = g.es[i]
        source_cluster = mem[edge.source]
        target_cluster = mem[edge.target]
        inter_cluster_edges[source_cluster, target_cluster] += (source_cluster != target_cluster)
    return inter_cluster_edges

def get_larger_partitions(partition_mem_arr, inter_cluster_edges):
    part_dict = {}
    unique_clusters = np.unique(partition_mem_arr)
    for i in unique_clusters:
        neighboring_partitions = pd.Series(partition_mem_arr).isin(np.where(inter_cluster_edges[i] > np.mean(inter_cluster_edges[i]))[0])
        part_dict[i] = neighboring_partitions
    return part_dict

print("Find inter_cluster_edges")
ic_edges = inter_cluster_edge_calc(partition, g, weighted = False)
ic_edges = ic_edges + ic_edges.T

np.savez(f'{out_dir}/ic_edges_all_counts.npz', ic_edges = ic_edges)

print("Find UMAP")
from umap import UMAP

# ic_edges = np.load('ic_edges_all_counts.npz')['ic_edges']

init = 'spectral'
min_dist = 0.1
n_neighbors = 45

def my_umap(mat, n_epochs, init=init, metric="cosine", repulsion_strength = 1, learning_rate = 1):
    reducer = UMAP(n_components = 2,
                   metric = metric,
                   spread = 1.0,
                   random_state = None,
                   learning_rate = learning_rate,
                   repulsion_strength = 1,
                   verbose = True,
                   # precomputed_knn = (knn_indices, knn_dists),
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   n_epochs = n_epochs,
                   init = init,
                   n_jobs = -1
                  )
    embedding = reducer.fit_transform(np.log1p(mat))
    return(embedding)


start = time.time()
mem_embeddings = my_umap(ic_edges, n_epochs = 2000, metric = 'cosine')
mem_embeddings[:, 0] -= np.mean(mem_embeddings[:, 0])
mem_embeddings[:, 1] -= np.mean(mem_embeddings[:, 1])
end = time.time()

plt.scatter(mem_embeddings[:, 0], mem_embeddings[:, 1], s = 1)
plt.title(f'New SB1 selection')
plt.savefig(f'{out_dir}/mem_embeddings_all_counts.png')
np.savez(f'{out_dir}/mem_embeddings_all_counts.npz', embeddings = mem_embeddings)

