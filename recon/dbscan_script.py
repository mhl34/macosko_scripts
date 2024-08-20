import numpy as np
import scipy
from helpers import *
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import pickle
import csv
import multiprocessing

unplaced = 0
total = 0

num_cores_os = os.cpu_count()
print(f"Number of CPU cores (os): {num_cores_os}")

def dbscan_location(embedding, matrix, col_index, total, unplaced, min_samples = 5, eps = 1, n_jobs = -1):
    idx = mat[:, col_index].nonzero()[0]
    embeddings_nonzero = embedding[idx]
    weights = mat[idx, col_index].data
    clusters = DBSCAN(min_samples = 10, eps = 0.3, n_jobs = n_jobs).fit(embeddings_nonzero, sample_weight = weights)
    filtered_embeddings = embeddings_nonzero[clusters.labels_ == 0]
    if len(set(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0) != 1:
        print(f"multiple clusters for {col_index}")
        unplaced += 1
        return col_index, -10000, -10000
    x, y = (filtered_embeddings[:, 0].mean(), filtered_embeddings[:, 1].mean())
    total += 1
    return col_index, x, y

embedding = np.load('min_umis_4/1/embedding_mat.npz')['embeddings']
mat = scipy.sparse.load_npz('min_umis_4/1/intermediate_files/mat.npz')

with open("sb1_embeddings_stricter.csv", "wb") as f:
    for i in tqdm(range(mat.shape[1])):
        col_index, x, y = dbscan_location(embedding, mat, i, total, unplaced)
        f.write('{0},{1},{2}\n'.format(col_index, x, y).encode('utf-8'))

print(f'Finished with {unplaced} unplaced beads out of {total} total beads, or {unplaced / total}')
