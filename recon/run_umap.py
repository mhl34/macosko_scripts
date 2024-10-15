import scipy
import argparse
import numpy as np
import cupy as cp
from helpers import *
from umap import UMAP

parser = argparse.ArgumentParser(description='Parser for KNN')

parser.add_argument('-i', '--in_dir', default='.', dest='in_dir')      # option that takes a value
parser.add_argument('-o', '--out_dir', default='.', dest='out_dir')
parser.add_argument('-n', '--n_neighbors', default=45, dest='n_neighbors')
parser.add_argument('-e', '--epochs', default=1000, dest='n_epochs')

args = parser.parse_args()

in_dir = args.in_dir
out_dir = args.out_dir
n_neighbors = int(args.n_neighbors)
n_epochs = int(args.n_epochs)

print('load mat')
mat = scipy.sparse.load_npz(f'{in_dir}/mat.npz')
print(f'mat shape: {mat.shape}')

print('load init')
mem = np.load(f'{in_dir}/membership_all_counts.npz')['membership']
mem_embeddings = np.load(f'{in_dir}/mem_embeddings_all_counts.npz')['embeddings']
init = mem_embeddings[mem]
print(f'init shape: {init.shape}')
min_dist = 0.1

print('load knn')
knn_output = np.load('knn.npz')
knn_indices = knn_output['indices'][:, :n_neighbors]
knn_dists = knn_output['dists'][:, :n_neighbors]

print('params')
print(f'neighbors: {n_neighbors}')
print(f'epochs: {n_epochs}')

def my_umap(mat, n_epochs, init=init, metric="cosine", repulsion_strength = 1, learning_rate = 1):
    reducer = UMAP(n_components = 2,
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

print('run umap')
embeddings = my_umap(mat, n_epochs)

print('save embeddings')
np.savez(f'{out_dir}/embeddings.npz', embeddings = embeddings)
