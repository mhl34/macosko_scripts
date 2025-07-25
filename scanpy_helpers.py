import scanpy as sc
import episcanpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.metrics import pairwise_distances

# in place function for adding the silhouette score to an object
def add_silhouette_score(adata, label, red_label = 'PCA', metric = 'euclidean', n_jobs = -1):
    mat = np.array(adata.obsm['PCA']).copy()
    labels = adata.obs[label]
    distances = pairwise_distances(mat, metric="euclidean", n_jobs=32)  # or 'euclidean'
    sil_samples = silhouette_samples(distances, labels, metric="precomputed")
    adata.obs[f'silhouette_samples_{label}'] = sil_samples
