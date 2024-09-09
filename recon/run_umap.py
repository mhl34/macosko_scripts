from cuml.manifold.umap import UMAP as cuUMAP
import scipy
import argparse
import numpy as np
import cupy as cp
from helpers import *

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--knn", help="knn type", type=str)
parser.add_argument("-g", "--gpu_id", help="gpu id", type=int, default = 0)
args, unknown = parser.parse_known_args()

knn = args.knn
gpu_id = args.gpu_id
mat = scipy.sparse.load_npz('mat.npz')
n_epochs = 100000
init = 'spectral'
n_neighbors = 45
min_dist = 0.1

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

print(f'load in knn of type : {knn}')
if knn == 'cuknn':
    cuknn_output = np.load('cuknn_output.npz')
    knn_indices = cuknn_output['knn_indices']
    knn_dists = cuknn_output['knn_dists']
elif knn == 'knn': 
    knn_output = np.load('knn_output.npz')
    knn_indices = knn_output['knn_indices']
    knn_dists = knn_output['knn_dists']
else:
    knn_output_150 = np.load('knn_output_150.npz')
    knn_indices = knn_output_150['knn_indices']
    knn_dists = knn_output_150['knn_dists']

# print('run mnn')
# connectivity = 'full_tree'
# knn_indices, knn_dists = mutual_nn_nearest(knn_indices, knn_dists, n_neighbors, n_neighbors, connectivity)

print('run umap')
embeddings = my_cuumap(mat, n_epochs)

print('save embeddings')
np.savez(f'embeddings_{knn}.npz', embeddings = embeddings)
