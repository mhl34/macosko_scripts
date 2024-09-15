import cuml
from cuml.manifold.umap import UMAP as cuUMAP
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import imageio
from umap import UMAP
import os
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tqdm import tqdm
import pickle
from scipy import special
import numba
import gc
import argparse
from helpers import *
import csv
from scipy import stats

# parameters
min_dist = 0.1
init = 'spectral'
n_epochs = 100000

def cuknn_descent(mat, n_neighbors, metric="cosine"):
    """
    creating nearest neighbors
    """
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    knn_cuml = cuNearestNeighbors(n_neighbors = n_neighbors,
                                    metric = metric,
                                    verbose = True,
                                    output_type = 'array'
                                 )
    knn_cuml.fit(mat)
    knn_dists, knn_indices = knn_cuml.kneighbors(mat, n_neighbors)
    return knn_indices, knn_dists

def hexmap(embedding, plot_name = None):
    """
    plotting with overlaps
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    x, y = embedding[:, 0], embedding[:, 1]
    hb = ax.hexbin(x, y, cmap='viridis')
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if plot_name:
        plt.savefig(f"{plot_name}.png")
        
def my_umap(mat, n_epochs, init=init, metric="cosine", repulsion_strength = 1, learning_rate = 1):
    reducer = UMAP(n_components = 2,
                   metric = metric,
                   spread = 1.0,
                   random_state = 0,
                   learning_rate = learning_rate,
                   repulsion_strength = repulsion_strength,
                   verbose = True,
                   precomputed_knn = (knn_indices, knn_dists),
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   n_epochs = n_epochs,
                   init = init
                  )
    embedding = reducer.fit_transform(np.log1p(mat))
    return(embedding)
        
def my_cuumap(mat, n_epochs, init=init, metric="cosine", repulsion_strength = 1, learning_rate = 1):
    reducer = cuUMAP(n_components = 2,
                   metric = metric,
                   spread = 1.0,
                   random_state = None,
                   learning_rate = learning_rate,
                   repulsion_strength = repulsion_strength,
                   verbose = True,
                   precomputed_knn = (knn_indices, knn_dists),
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   n_epochs = n_epochs,
                   init = init
                  )
    embedding = reducer.fit_transform(np.log1p(mat))
    return(embedding)

def create_knn_matrix(knn_indices, knn_dists, n_neighbors):
    assert knn_indices.shape == knn_dists.shape
    assert n_neighbors <= knn_indices.shape[1]
    rows = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.int32)
    cols = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.int32)
    vals = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.float32)
    
    pos = 0
    for i, indices in enumerate(knn_indices):
        for j, index in enumerate(indices[:n_neighbors]):
            if index == -1:
                continue
            rows[pos] = i 
            cols[pos] = index
            vals[pos] = knn_dists[i][j]
            pos += 1
    
    knn_matrix = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))
    return knn_matrix

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dropout', dest='dropout', help='dropout rate')      # option that takes a value
parser.add_argument('-mnn', action='store_true', dest='mnn', help='use mnn') # use
parser.add_argument('-n', '--n_neighbors', dest = 'n_neighbors', default = 45, help='number of neighbors')
parser.add_argument('-c', action='store_true', dest='cache', help='use previous run')
parser.add_argument('-lt', '--low_threshold', dest='low', default = 10, help='low number of connections')
parser.add_argument('-ht', '--high_threshold', dest='high', default = 3000, help='high number of connections')
parser.add_argument('-rd_sb1', dest='rd_sb1', default = 0, help='random dropout of x percentage of sb1 beads')
parser.add_argument('-rd_sb2', dest='rd_sb2', default = 0, help='random dropout of x percentage of sb2 beads')

args = parser.parse_args()
dropout = args.dropout
mnn = args.mnn
n_neighbors = args.n_neighbors
cache = args.cache
low = args.low
high = args.high
rd_sb1 = float(args.rd_sb1)
rd_sb2 = float(args.rd_sb2)

l1 = int(low)
l2 = int(low)
h1 = int(high)
h2 = int(high)

print('filter matrix')
df = pd.read_csv(f'{dropout}/matrix.csv.gz', compression='gzip')
df.sb1_index -= 1 # convert from 1- to 0-indexed
df.sb2_index -= 1 # convert from 1- to 0-indexed
sb1 = pd.read_csv(f'{dropout}/sb1.csv.gz', compression='gzip')
sb2 = pd.read_csv(f'{dropout}/sb2.csv.gz', compression='gzip')
df, uniques1, uniques2, _, _ = connection_filter(df)
mat = coo_matrix((df['umi'], (df['sb2_index'], df['sb1_index']))).tocsr()

# scipy.sparse.save_npz(f"{dropout}/mat.npz", mat)
    
# connectivity = "full_tree"

# # print('cuknn descent')
# # knn_indices, knn_dists = cuknn_descent(np.log1p(mat), n_neighbors, metric = "cosine")
# print('knn descent (150)')
# n_neighbors = 45
# n_neighbors2 = 150
# knn_indices, knn_dists = knn_descent(np.log1p(mat), n_neighbors2, metric = "cosine")
# knn_indices = knn_indices[:, :45]
# knn_dists = knn_dists[:, :45]
# with open(f'{dropout}/knn_150_output_{n_epochs}_{dropout}.npz', 'wb') as f:
#     np.savez(f, knn_indices = knn_indices, knn_dists = knn_dists)
# with open(f'{dropout}/intermediate_files/knn_output_rd_sb1_{rd_sb1}_rd_sb2_{rd_sb2}.npz', 'wb') as f:
#     np.savez(f, knn_indices = knn_indices, knn_dists = knn_dists)

# with open(f'{dropout}/knn_output_cuknn_{dropout}.npz', 'wb') as f:
#     np.savez(f, knn_indices = knn_indices, knn_dists = knn_dists)

print('load mutual neighbors')
# knn_indices, knn_dists =  mutual_nn_nearest(knn_indices, knn_dists, n_neighbors, n_neighbors)
# with open(f'{dropout}/mnn_output_{n_epochs}_150_{dropout}.npz', 'wb') as f:
#     np.savez(f, mnn_indices = knn_indices, mnn_dists = knn_dists)

knn_output = np.load(f'{dropout}/mnn_output_{n_epochs}_150_{dropout}.npz')
knn_indices = knn_output['mnn_indices']
knn_dists = knn_output['mnn_dists']

print('umap')
init = "spectral"
embeddings = my_cuumap(mat, n_epochs, init=init)

with open(f'{dropout}/embedding_mat_mnn_150_{n_epochs}_{dropout}.npz', 'wb') as f:
    np.savez(f, embeddings = embeddings)

# hexmap(embeddings, f"{dropout}/umap_mnn_{n_epochs}_cuknn" if mnn else f"{dropout}/umap_{n_epochs}_cuknn")

sbs = [sb2["sb2"][i] for i in uniques2]
assert embedding.shape[0] == len(sbs)
with open(os.path.join(f"{dropout}/Puck_{n_epochs}_cuknn.csv"), mode='w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(sbs)):
        writer.writerow([sbs[i], embeddings[i,0], embeddings[i,1]])

print("\nDone!")
    
# hexmap(embeddings, f"{dropout}/outputs/umap_{n_epochs}_{connectivity}_rd_sb1_{rd_sb1}_rd_sb2_{rd_sb2}" if mnn else f"{dropout}/outputs/umap_{n_epochs}_knn_rd_sb1_{rd_sb1}_rd_sb2_{rd_sb2}")

