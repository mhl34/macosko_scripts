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

# process anndata object
def process(adata, batch_key = 'donor_id', harmony_key = None, regress_key = None, n_top_genes = 2000, n_jobs = 16):
    print('normalize...')
    sc.pp.normalize_total(adata)
    print('scale counts...')
    sc.pp.scale(adata)
    if regress_key != None:
        print('regress out...')
        sc.pp.regress_out(adata, regress_key, n_jobs = n_jobs)
    print('highly variable genes...')
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key)
    print('pca...')
    sc.tl.pca(adata)
    if harmony_key != None:
        print('run harmony...')
        sc.external.pp.harmony_integrate(adata, harmony_key)
    print('find neighbors...')
    sc.pp.neighbors(adata)
    print('run umap...')
    sc.tl.umap(adata)
