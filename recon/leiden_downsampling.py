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

dropouts = [1, 0.8, 0.6, 0.4, 0.2]
idx = 0

for dropout in dropouts:
    print(f'KNN output: {dropout}')
    mnn_output = np.load(f'{dropout}/knn_output.npz')
    mnn_indices = mnn_output['knn_indices']
    mnn_dists = mnn_output['knn_dists']
    knn_indices = mnn_indices
    knn_dists = mnn_dists
    
    # Check the shape and dimensions
    print("KNN indices shape:", knn_indices.shape)
    print("KNN distances shape:", knn_dists.shape)
    
    # Create graph edges with distances
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
    
    np.savez(f'membership_{dropout}.npz', membership = np.array(partition.membership))
    
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
    
    np.savez(f'ic_edges_{dropout}.npz', ic_edges = ic_edges)
    
    print("Find UMAP")
    from umap import UMAP
    
    init = 'spectral'
    min_dist = 0.01
    n_neighbors = 45
    
    def my_umap(mat, n_epochs, init=init, metric="cosine", repulsion_strength = 1, learning_rate = 1):
        reducer = UMAP(n_components = 2,
                       metric = metric,
                       spread = 1.0,
                       random_state = None,
                       learning_rate = learning_rate,
                       repulsion_strength = 2,
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
    mem_embeddings = my_umap(ic_edges, n_epochs = 2000, metric = 'correlation')
    mem_embeddings[:, 0] -= np.mean(mem_embeddings[:, 0])
    mem_embeddings[:, 1] -= np.mean(mem_embeddings[:, 1])
    end = time.time()

    plt.figure(idx)
    plt.scatter(mem_embeddings[:, 0], mem_embeddings[:, 1], s = 1)
    plt.title(f'dropout: {dropout}')
    plt.savefig(f'mem_embeddings_{dropout}.png')
    np.savez(f'mem_embeddings_{dropout}.npz', embeddings = mem_embeddings)
    idx += 1

