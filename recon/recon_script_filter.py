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

# functions
def knn_descent(mat, n_neighbors, metric="cosine", n_cores=-1, metric_kwds = {}):
    """
    creating nearest neighbors
    """
    from umap.umap_ import nearest_neighbors
    knn_indices, knn_dists, _ = nearest_neighbors(mat,
                                    n_neighbors = n_neighbors,
                                    metric = metric,
                                    metric_kwds = metric_kwds,
                                    angular = False, # Does nothing?
                                    random_state = None, # sklearn.utils.check_random_state(0)
                                    low_memory = True, # False?
                                    use_pynndescent = True, # Does nothing?
                                    n_jobs = n_cores,
                                    verbose = True
                                )
    return knn_indices, knn_dists

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

df = pd.read_csv(f'{dropout}/matrix.csv.gz', compression='gzip')
df.sb1_index -= 1 # convert from 1- to 0-indexed
df.sb2_index -= 1 # convert from 1- to 0-indexed
sb1 = pd.read_csv(f'{dropout}/sb1.csv.gz', compression='gzip')
sb2 = pd.read_csv(f'{dropout}/sb2.csv.gz', compression='gzip')
assert sorted(list(set(df.sb1_index))) == list(range(sb1.shape[0]))
assert sorted(list(set(df.sb2_index))) == list(range(sb2.shape[0]))
print(f"{sb1.shape[0]} R1 barcodes")
print(f"{sb2.shape[0]} R2 barcodes")
print("\nFiltering the beads...")
umi_before = sum(df["umi"])
sb1_low  = np.where(sb1['connections'] <  l1)[0]
sb2_low  = np.where(sb2['connections'] <  l2)[0]
sb1_high = np.where(sb1['connections'] >= h1)[0]
sb2_high = np.where(sb2['connections'] >= h2)[0]
rand_dropout_sb1 = np.random.choice(len(sb1), round(len(sb1) * rd_sb1))
rand_dropout_sb2 = np.random.choice(len(sb2), round(len(sb2) * rd_sb2))
unplaced_sb2 = np.load(f'unplaced/unplaced_{dropout}.npz')['arr']
print(f"{len(sb1_low)} low R1 beads filtered ({len(sb1_low)/len(sb1)*100:.2f}%)")
print(f"{len(sb2_low)} low R2 beads filtered ({len(sb2_low)/len(sb2)*100:.2f}%)")
print(f"{len(sb1_high)} high R1 beads filtered ({len(sb1_high)/len(sb1)*100:.2f}%)")
print(f"{len(sb2_high)} high R2 beads filtered ({len(sb2_high)/len(sb2)*100:.2f}%)")
print(f"{len(rand_dropout_sb1)} random R1 beads filtered ({len(rand_dropout_sb1)/len(sb1)*100:.2f}%)")
print(f"{len(rand_dropout_sb2)} random R2 beads filtered ({len(rand_dropout_sb2)/len(sb2)*100:.2f}%)")
df = df[~df['sb1_index'].isin(sb1_low) 
& ~df['sb1_index'].isin(sb1_high) 
& ~df['sb2_index'].isin(sb2_low) 
& ~df['sb2_index'].isin(sb2_high) 
& ~df['sb1_index'].isin(rand_dropout_sb1) 
& ~df['sb2_index'].isin(rand_dropout_sb2)]
umi_after = sum(df["umi"])
print(f"{umi_before-umi_after} UMIs filtered ({(umi_before-umi_after)/umi_before*100:.2f}%)")
codes1, uniques1 = pd.factorize(df['sb1_index'], sort=True)
df.loc[:, 'sb1_index'] = codes1
codes2, uniques2 = pd.factorize(df['sb2_index'], sort=True)
df.loc[:, 'sb2_index'] = codes2
assert sorted(list(set(df.sb1_index))) == list(range(len(set(df.sb1_index))))
assert sorted(list(set(df.sb2_index))) == list(range(len(set(df.sb2_index))))

mat = coo_matrix((df['umi'], (df['sb2_index'], df['sb1_index']))).tocsr()

# scipy.sparse.save_npz(f"{dropout}/intermediate_files/mat.npz", mat)
    
connectivity = "full_tree"
    
knn_indices, knn_dists = cuknn_descent(np.log1p(mat), n_neighbors, metric = "cosine")
# with open(f'{dropout}/intermediate_files/knn_output_rd_sb1_{rd_sb1}_rd_sb2_{rd_sb2}.npz', 'wb') as f:
#     np.savez(f, knn_indices = knn_indices, knn_dists = knn_dists)

with open(f'{dropout}/knn_output_unplaced_filter_{dropout}.npz', 'wb') as f:
    np.savez(f, knn_indices = knn_indices, knn_dists = knn_dists)

knn_indices, knn_dists = mutual_nn_nearest(np.log1p(mat), n_neighbors, metric = "cosine")
with open(f'{dropout}/mnn_output_unplaced_filter_{dropout}.npz', 'wb') as f:
    np.savez(f, mnn_indices = knn_indices, mnn_dists = knn_dists)

# default learning rate: 1.0
init = "spectral"
embeddings = my_cuumap(mat, n_epochs, init=init, learning_rate = 1, repulsion_strength = 2)

# with open(f'{dropout}/outputs/embedding_mat_rd_sb1_{rd_sb1}_rd_sb2_{rd_sb2}.npz', 'wb') as f:
#     np.savez(f, embeddings = embeddings)

with open(f'{dropout}/embedding_mat_unplaced_filter.npz', 'wb') as f:
    np.savez(f, embeddings = embeddings)

# with open(os.path.join(f"{dropout}/outputs/Puck_rd_sb1_{rd_sb1}_rd_sb2_{rd_sb2}.csv"), mode='w', newline='') as file:
#     writer = csv.writer(file)
#     for i in range(len(sbs)):
#         writer.writerow([sbs[i], embeddings[i,0], embeddings[i,1]])

# print("\nDone!")

with open(os.path.join(f"{dropout}/Puck_unplaced_filter.csv"), mode='w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(sbs)):
        writer.writerow([sbs[i], embeddings[i,0], embeddings[i,1]])

print("\nDone!")
    
# hexmap(embeddings, f"{dropout}/outputs/umap_{n_epochs}_{connectivity}_rd_sb1_{rd_sb1}_rd_sb2_{rd_sb2}" if mnn else f"{dropout}/outputs/umap_{n_epochs}_knn_rd_sb1_{rd_sb1}_rd_sb2_{rd_sb2}")

hexmap(embeddings, f"{dropout}/umap_{connectivity}_{n_epochs}_unplaced_filter" if mnn else f"{dropout}/umap_{n_epochs}_unplaced_filter")
