import os
import math
import copy
import heapq
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from functools import reduce
from collections import Counter
import gc
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn, zip_sp_matmul_topn
import leidenalg as la
import igraph as igp
from umap import UMAP

# (x, y)
def hexmap(embedding, title="", fontsize=12):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x, y = embedding[:, 0], embedding[:, 1]
    hb = ax.hexbin(x, y, cmap='viridis', linewidths=0.5)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{title}', fontsize=fontsize)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig, ax

# (x, y)
def weighted_hexmap(embedding, weights, title="", fontsize=12):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x, y = embedding[:, 0], embedding[:, 1]
    hb = ax.hexbin(x, y, cmap='inferno', C = weights, linewidths=0.5)
    fig.colorbar(hb, ax = ax)
    ax.axis('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{title}', fontsize=fontsize)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig, ax

# [(x, y)]
def hexmaps(embeddings, titles=[], fontsize=10):
    assert type(embeddings) == type(titles) == list
    n = math.ceil(len(embeddings)**0.5)
    fig, axes = plt.subplots(n, n, figsize=(8, 8))
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    axes = axes.flatten()

    if len(titles) == 0:
        titles = ["" for _ in range(len(embeddings))]
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    assert len(titles) == len(embeddings)
    
    for ax, embedding, title in zip(axes, embeddings, titles):
        x, y = embedding[:, 0], embedding[:, 1]
        hb = ax.hexbin(x, y, cmap='viridis', linewidths=0.5)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{title}', fontsize=fontsize)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
    for ax in axes[len(embeddings):]:
        ax.set_visible(False)

    return fig, axes

# (x, y, color)
def beadplot(embedding, colors, cmap='viridis'):
    assert embedding.shape[0] == len(colors)
    if isinstance(embedding, np.ndarray):
        puck = np.hstack((embedding, colors.reshape(-1,1)))
        puck = puck[puck[:,2].argsort()]
        x = puck[:,0]
        y = puck[:,1]
        c = puck[:,2]
    elif isinstance(embedding, pd.DataFrame):
        puck = embedding
        puck.loc[:,"color"] = colors
        puck = puck.sort_values(by=puck.columns[2])
        x = puck.iloc[:,0]
        y = puck.iloc[:,1]
        c = puck.iloc[:,2]
    else:
        raise TypeError("Input must be a NumPy ndarray or a pandas DataFrame")

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c=c, cmap=cmap, s=0.1)
    plt.colorbar()
    plt.xlabel('xcoord')
    plt.ylabel('ycoord')
    plt.axis('square')
    return plt

# embedding1, embedding2
def L2_distance(p1, p2):
    assert p1.shape == p2.shape
    dists = np.sqrt(np.sum(np.square(p1 - p2), axis=1))
    return np.sum(dists) / p1.shape[0]

# embedding1, embedding2
def procrustes_distance(p1, p2):
    assert p1.shape == p2.shape
    from scipy.spatial import procrustes
    _, _, disparity = procrustes(p1, p2)
    return disparity

# [embedding]
def convergence_plot(embeddings):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    x = [(i+2)*1000 for i in range(len(embeddings)-1)]

    assert type(embeddings) == list
    if len(embeddings) < 2:
        return fig, axes

    y1 = [L2_distance(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]
    axes[0].scatter(x, y1, color='blue')
    axes[0].plot(x, y1, color='red')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Mean UMAP Distance')
    axes[0].set_title('Average L2 Displacement')

    y2 = [procrustes_distance(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]
    axes[1].scatter(x, y2, color='blue')
    axes[1].plot(x, y2, color='red')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Disparity')
    axes[1].set_title('Procrustes Disparity')
    
    plt.tight_layout()
    return fig, axes

# highlight reciprocated ones
def embedding_neighbor_locations(knn_indices, knn_dists, embedding, nn=45, n=16):
    from matplotlib.cm import viridis
    from matplotlib.colors import Normalize
    assert knn_indices.shape[0] == knn_dists.shape[0] == embedding.shape[0]
    assert knn_indices.shape[1] == knn_dists.shape[1]
    
    # Create the grid
    nrows = np.ceil(np.sqrt(n)).astype(int)
    fig, axes = plt.subplots(nrows, nrows, figsize=(8, 8))
    if type(axes) != np.ndarray:
        axes = np.array([axes])
    axes = axes.flatten()

    selected_indices = np.random.randint(0, embedding.shape[0], size=n)
    x = embedding[:,0] ; y = embedding[:,1]

    # Plot the data
    for ax, i in zip(axes, selected_indices):
        indices = knn_indices[i, 1:nn]
        dists = knn_dists[i, 1:nn]
        colors = viridis(Normalize(vmin=0, vmax=1)(dists))

        ax.scatter(x, y, color='grey', s=10, alpha=0.5)
        ax.scatter(x[indices], y[indices], color=colors, s=20)
        ax.scatter(x[i], y[i], color='red', s=30)

        ax.set_xlim(min(x[indices]), max(x[indices]))
        ax.set_ylim(min(y[indices]), max(y[indices]))
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(i)
    
    for ax in axes[n:]:
        ax.set_visible(False)
    
    fig.tight_layout()
    return fig, axes

def embedding_neighbor_distances(knn_indices, knn_dists, embedding, nn=45):
    assert knn_indices.shape[0] == knn_dists.shape[0] == embedding.shape[0]
    assert knn_indices.shape[1] == knn_dists.shape[1]

    dists = np.vstack([np.linalg.norm(embedding[inds] - embedding[inds[0]], axis=1) for inds in knn_indices])
    
    def n_hist(ax, dists, nn):
        data = dists[:,nn]
        ax.hist(np.log10(data), bins=100)
        ax.set_xlabel('UMAP distance (log10)')
        ax.set_ylabel('Count')
        ax.set_title(f'UMAP Distance to neighbor {nn}')
        meanval = np.log10(np.mean(data))
        ax.axvline(meanval, color='red', linestyle='dashed')
        ax.text(meanval+0.1, ax.get_ylim()[1] * 0.95, f'Mean: {10**meanval:.2f}', color='black', ha='left')

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    n_hist(axes[0,0], dists, 1)
    n_hist(axes[0,1], dists, nn)
    
    ax=axes[1,0]
    data = np.mean(dists[:,1:nn], axis=1)
    ax.hist(np.log10(data), bins=100)
    ax.set_xlabel('UMAP distance (log10)')
    ax.set_ylabel('Count')
    ax.set_title(f'UMAP Distance to average neighbor (45)')
    meanval = np.log10(np.mean(data))
    ax.axvline(meanval, color='red', linestyle='dashed')
    ax.text(meanval+0.1, ax.get_ylim()[1] * 0.95, f'Mean: {10**meanval:.2f}', color='black', ha='left')
    
    axes[1,1].hexbin(np.log10(dists[:,1:nn]), knn_dists[:,1:nn], gridsize=100, bins='log', cmap='plasma')
    axes[1,1].set_xlabel('UMAP distance (log10)')
    axes[1,1].set_ylabel('Cosine Distance')
    axes[1,1].set_title(f'Cosine vs. UMAP Distance ({nn})')
    
    fig.tight_layout()
    
    return fig, axes

def umi_density_plot(puck_file, sb2_file):
    puck = pd.read_csv(puck_file, names = ['sb2', 'x', 'y'])
    sb2 = pd.read_csv(sb2_file)
    merged = pd.merge(puck, sb2, on = 'sb2')

    # get log UMI density plot
    embedding = np.array([(x,y) for x,y in zip(merged.x, merged.y)])
    weights = np.log1p(merged.umi)
    fig, ax = weighted_hexmap(embedding, weights, title="Log UMI Density", fontsize = 12)
    return fig, ax

### BEAD FILTERING METHODS #####################################################
def connection_filter(df):
    assert all(df.columns == ["sb1_index", "sb2_index", "umi"])
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    meta = {"umi_init": sum(df["umi"])}

    def hist_z(ax, data, z_high=None, z_low=None):
        ax.hist(data, bins=100)
        ax.set_ylabel('Count')
    
        meanval = np.mean(data)
        ax.axvline(meanval, color='black', linestyle='dashed')
        ax.text(meanval, ax.get_ylim()[1] * 0.95, f'Mean: {10**meanval:.2f}', color='black', ha='center')
    
        if z_high:
            assert z_high > 0
            zval = meanval + np.std(data) * z_high
            ax.axvline(zval, color='red', linestyle='dashed')
            ax.text(zval, ax.get_ylim()[1]*0.9, f'{z_high}z: {10**zval:.2f}', color='red', ha='left')
    
        if z_low:
            assert z_low < 0
            zval = meanval + np.std(data) * z_low
            ax.axvline(zval, color='red', linestyle='dashed')
            ax.text(zval, ax.get_ylim()[1]*0.9, f'{z_low}z: {10**zval:.2f}', color='red', ha='right')
    
    # Remove sb1 beads
    z_high = 3
    sb1 = df.groupby(f'sb1_index').agg(umi=('umi', 'sum'), connections=('umi', 'size'), max=('umi', 'max')).reset_index()
    hist_z(axes[0,0], np.log10(sb1['connections']), z_high)
    axes[0,0].set_xlabel('Connections')
    axes[0,0].set_title('sb1 connections')
    hist_z(axes[0,1], np.log10(sb1['max']))
    axes[0,1].set_xlabel('Max')
    axes[0,1].set_title('sb1 max')

    logcon = np.log10(sb1['connections'])
    high = np.where(logcon >= np.mean(logcon) + np.std(logcon) * z_high)[0]
    sb1_remove = reduce(np.union1d, [high])
    df = df[~df['sb1_index'].isin(sb1_remove)]
    
    meta["sb1_high"] = len(high)
    meta["sb1_removed"] = len(sb1_remove)
    meta["umi_half"] = sum(df["umi"])
    print(f"{len(high)} high R1 beads ({len(high)/len(sb1)*100:.2f}%)")
    diff = meta['umi_init']-meta['umi_half'] ; print(f"{diff} R1 UMIs filtered ({diff/meta['umi_init']*100:.2f}%)")

    # Remove sb2 beads
    z_high = 3 ; z_low = -3
    sb2 = df.groupby(f'sb2_index').agg(umi=('umi', 'sum'), connections=('umi', 'size'), max=('umi', 'max')).reset_index()
    hist_z(axes[1,0], np.log10(sb2['connections']), z_high, z_low)
    axes[1,0].set_xlabel('Connections')
    axes[1,0].set_title('sb2 connections')
    hist_z(axes[1,1], np.log10(sb2['max']))
    axes[1,1].set_xlabel('Max')
    axes[1,1].set_title('sb2 max')
    
    logcon = np.log10(sb2['connections'])
    high = np.where(logcon >= np.mean(logcon) + np.std(logcon) * z_high)[0]
    low = np.where(logcon <= np.mean(logcon) + np.std(logcon) * z_low)[0]
    noise = np.where(sb2['max'] <= 1)[0]
    sb2_remove = reduce(np.union1d, [high, low, noise])
    df = df[~df['sb2_index'].isin(sb2_remove)]
    
    meta["sb2_high"] = len(high)
    meta["sb2_low"] = len(low)
    meta["sb2_noise"] = len(noise)
    meta["sb2_removed"] = len(sb2_remove)
    meta["umi_final"] = sum(df["umi"])
    print(f"{len(high)} high R2 beads ({len(high)/len(sb2)*100:.2f}%)")
    print(f"{len(low)} low R2 beads ({len(low)/len(sb2)*100:.2f}%)")
    print(f"{len(noise)} noise R2 beads ({len(noise)/len(sb2)*100:.2f}%)")
    diff = meta['umi_half']-meta['umi_final'] ; print(f"{diff} R2 UMIs filtered ({diff/meta['umi_half']*100:.2f}%)")
    
    # Factorize the new dataframe
    codes1, uniques1 = pd.factorize(df['sb1_index'], sort=True)
    df.loc[:, 'sb1_index'] = codes1
    codes2, uniques2 = pd.factorize(df['sb2_index'], sort=True)
    df.loc[:, 'sb2_index'] = codes2

    # assert set(df.sb1_index) == set(range(max(df.sb1_index)+1))
    # assert set(df.sb1_index) == set(range(max(df.sb1_index)+1))
    # assert set(df.sb2_index) == set(range(max(df.sb2_index)+1))
    
    fig.tight_layout()
    return df, uniques1, uniques2, fig, meta


def knn_filter(knn_indices, knn_dists):
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    filter_indexes = set()
    meta = dict()

    def hist_z(ax, data, z_high=None, z_low=None, bins=100):
        ax.hist(data, bins=bins)
        ax.set_ylabel('Count')
    
        meanval = np.mean(data)
        ax.axvline(meanval, color='black', linestyle='dashed')
        ax.text(meanval, ax.get_ylim()[1] * 0.95, f'Mean: {meanval:.2f}', color='black', ha='center')
    
        if z_high:
            assert z_high > 0
            zval = meanval + np.std(data) * z_high
            ax.axvline(zval, color='red', linestyle='dashed')
            ax.text(zval, ax.get_ylim()[1]*0.9, f'{z_high}z: {zval:.2f}', color='red', ha='left')
    
        if z_low:
            assert z_low < 0
            zval = meanval + np.std(data) * z_low
            ax.axvline(zval, color='red', linestyle='dashed')
            ax.text(zval, ax.get_ylim()[1]*0.9, f'{z_low}z: {zval:.2f}', color='red', ha='right')
    
    # Filter beads with far nearest neighbors
    z_high = 3
    data = knn_dists[:,1]
    hist_z(axes[0,0], data, z_high)
    axes[0,0].set_xlabel(f'Distance')
    axes[0,0].set_title(f'Nearest neighbor distance')

    high = knn_indices[data >= np.mean(data) + np.std(data) * z_high, 0]
    filter_indexes.update(high)
    print(f"{len(high)} far-NN beads removed")
    meta["far-NN"] = len(high)

    # Plot furthest neighbor distance
    hist_z(axes[0,1], knn_dists[:,-1])
    axes[0,1].set_xlabel(f'Distance')
    axes[0,1].set_title(f'Furthest neighbor distance ({knn_dists.shape[1]})')
    
    # Filter too-high or too-low in-edges
    z_high = 3 ; z_low = -3
    indexes, data = np.unique(knn_indices, return_counts=True)
    hist_z(axes[1,0], data, z_high, z_low, bins=np.arange(0, data.max()+1))
    axes[1,0].set_xlabel('In-edges')
    axes[1,0].set_title('Number of in-edges')
    axes[1,0].set_xlim(0, 4*knn_indices.shape[1])
    
    high = indexes[data >= np.mean(data) + np.std(data) * z_high]
    filter_indexes.update(high)
    print(f"{len(high)} in-high beads removed")
    meta["in-high"] = len(high)
    
    low = indexes[data <= np.mean(data) + np.std(data) * z_low]
    filter_indexes.update(low)
    print(f"{len(low)} in-low beads removed")
    meta["in-low"] = len(low)

    # Filter low clustering coefficients
    z_low = -3
    knn_matrix = create_knn_matrix(knn_indices, knn_dists)
    G = nx.from_scipy_sparse_array(knn_matrix, create_using=nx.Graph, edge_attribute=None) # undirected, unweighted
    clustering = nx.clustering(G, nodes=None, weight=None)
    data = [clustering[key] for key in sorted(clustering.keys())]
    hist_z(axes[1,1], data, z_low=z_low)
    axes[1,1].set_xlabel('Clustering coefficient')
    axes[1,1].set_title('Local clustering coefficient')
    
    low = knn_indices[data <= np.mean(data) + np.std(data) * z_low, 0]
    filter_indexes.update(low)
    print(f"{len(low)} cluster-low beads removed")
    meta["cluster-low"] = len(low)

    # Filter weakly-connected components
    n_components, labels = sp.csgraph.connected_components(csgraph=knn_matrix, directed=True, connection='strong')
    wcc = np.where(labels != np.bincount(labels).argmax())[0]
    filter_indexes.update(wcc)
    print(f"{len(wcc)} WCC beads removed")
    meta["wcc"] = len(wcc)
     
    fig.tight_layout()
    return filter_indexes, fig, meta



### KNN METHODS ################################################################
# Calculate Top-N KNN
def top_n_mat(mat, top_n = 150):
    # L2 normalize by row
    mat_norm = normalize(mat, norm='l2', axis=1, copy=True)
    mat_norm = mat_norm.astype(np.float32)

    # Take dot product of each row (top-N results written to prod)
    prod = sp_matmul_topn(mat_norm, mat_norm.T, top_n=top_n, sort=True, idx_dtype = np.dtype('int64'), n_threads = -1)

    # make cosine distance and clean up
    new_data = 1 - prod.data
    new_data *= (new_data > 0).astype(np.int64)
    mat_comp = sp.csr_matrix((new_data, prod.indices, prod.indptr), shape=prod.shape)
    mat_comp.eliminate_zeros()
    del prod
    gc.collect()

    return mat_comp

def split_top_n_mat(mat, top_n = 150, chunks = 2):
    # L2 normalize by row
    mat_norm = normalize(mat, norm='l2', axis=1, copy=True)
    mat_norm = mat_norm.astype(np.float32)

    # 2a. Split the sparse matrices. Here A is split into three parts, and B into five parts.
    print('\nChunk data...')
    chunk_size = mat_norm.shape[0] // chunks
    end = mat_norm.shape[0] % chunks
    As = [mat_norm[i * chunk_size: (i+1) * chunk_size + end * (i == chunks - 1)] for i in range(chunks)]
    Bs = [mat_norm[i * chunk_size: (i+1) * chunk_size + end * (i == chunks - 1)] for i in range(chunks)]
    del mat_norm
    gc.collect()
    
    print('\nMultiply sub-matrices...')
    Cs = [[sp_matmul_topn(Aj, Bi.T, top_n=top_n, sort=True, idx_dtype = np.dtype('int64'), n_threads = -1) for Bi in Bs] for Aj in As]
    del As
    del Bs
    gc.collect()
    
    print('\nZip sub-matrices...')
    Czip = [zip_sp_matmul_topn(top_n=150, C_mats=Cis) for Cis in Cs]
    del Cs
    gc.collect()
    
    print('\nStack matrices...')
    prod = sp.vstack(Czip, dtype=np.float32)

    # make cosine distance and clean up
    new_data = 1 - prod.data
    new_data *= (new_data > 0).astype(np.int64)
    mat_comp = sp.csr_matrix((new_data, prod.indices, prod.indptr), shape=prod.shape)
    mat_comp.eliminate_zeros()
    del prod
    gc.collect()

    return mat_comp

# Compute the KNN using NNDescent
def knn_descent(mat, n_neighbors, metric="cosine", n_jobs=-1):    
    from pynndescent import NNDescent
    knn_search_index = NNDescent(
        data=mat,
        metric=metric,
        metric_kwds={},
        n_neighbors=n_neighbors,
        n_trees=64, # originally None
        # leaf_size=None,
        pruning_degree_multiplier=3.0, # originally 1.5
        diversify_prob=0.0, # originally 1.0
        # tree_init=True,
        # init_graph=None,
        # init_dist=None,
        random_state=None,
        low_memory=True, # originally False
        max_candidates=60, # originally None
        max_rptree_depth=999999, # originally 200
        n_iters=512, # originally None
        delta=0.0001, # originally 0.001
        n_jobs=n_jobs,
        # compressed=False,
        # parallel_batch_queries=False,
        verbose=True # originally False
    )
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    return knn_indices, knn_dists

# UMAP NNDescent wrapper
def nearest_neighbors(mat, n_neighbors, metric="cosine", n_jobs=-1):
    from umap.umap_ import nearest_neighbors
    knn_indices, knn_dists, _ = nearest_neighbors(mat,
                                    n_neighbors = n_neighbors,
                                    metric = metric,
                                    metric_kwds = {},
                                    angular = False, # Does nothing?
                                    random_state = None,
                                    low_memory = True, # False?
                                    use_pynndescent = True, # Does nothing?
                                    n_jobs = n_jobs,
                                    verbose = True
                                )
    return knn_indices, knn_dists

# knn_indices1 can be row-sliced, knn_indices2 is original
def knn_merge(knn_indices1, knn_dists1, knn_indices2, knn_dists2):
    assert knn_indices1.shape == knn_dists1.shape
    assert knn_indices2.shape == knn_dists2.shape
    assert knn_indices1.shape[0] == knn_dists1.shape[0] == knn_indices2.shape[0] == knn_dists2.shape[0]

    assert all(knn_indices2[:,0] == np.arange(knn_indices2.shape[0]))
    assert np.all(knn_indices2 >= 0) and np.all(knn_dists2 >= 0)
    index_map = dict(zip(knn_indices1[:,0], knn_indices2[:,0]))
    knn_indices1 = np.vectorize(lambda i: index_map.get(i,-1))(knn_indices1)
    assert all(knn_indices1[:,0] == knn_indices2[:,0])
    # assert np.all(knn_dists1[np.where(knn_indices1==knn_indices2)] == knn_dists2[np.where(knn_indices1==knn_indices2)])
    
    k = max(knn_indices1.shape[1], knn_indices2.shape[1])
    knn_indices = np.zeros((knn_indices1.shape[0], k), dtype=np.int32)
    knn_dists = np.zeros((knn_indices1.shape[0], k), dtype=np.float64)
    
    for i in range(knn_indices1.shape[0]):
        d1 = {i:d for i,d in zip(knn_indices1[i], knn_dists1[i]) if i >= 0}
        d2 = {i:d for i,d in zip(knn_indices2[i], knn_dists2[i]) if i >= 0}
        d = d1 | d2

        row = sorted(d.items(), key=lambda item: item[1])[:k]
        inds, dists = zip(*row)
        
        knn_indices[i,:] = inds
        knn_dists[i,:] = dists

    return knn_indices, knn_dists

class KNNMask:
    def __init__(self, knn_indices, knn_dists):
        assert knn_indices.shape == knn_dists.shape
        self.knn_indices = copy.deepcopy(knn_indices)
        self.knn_dists = copy.deepcopy(knn_dists)
        self.original_size = knn_indices.shape[0]
        self.valid_original_indexes = np.arange(self.original_size)
    
    def remove(self, bad):
        curr_len = self.knn_indices.shape[0]
        
        assert self.knn_indices.shape == self.knn_dists.shape
        assert np.all(np.unique(bad, return_counts=True)[1] == 1)
        assert np.min(bad) >= 0 and np.max(bad) < curr_len
        print(f"Removing {len(bad)} beads")
        
        # Slice the data        
        m = np.ones(curr_len, dtype=bool)
        m[bad] = False
        self.knn_indices = self.knn_indices[m]
        self.knn_dists = self.knn_dists[m]
        self.valid_original_indexes = self.valid_original_indexes[m]
        assert np.all(b not in self.knn_indices[:,0] for b in bad)
        
        # Remap the KNN        
        index_map = np.cumsum([i not in bad for i in range(curr_len)], dtype=np.int32) - 1
        index_map[bad] = -1
        mask_2d = np.isin(self.knn_indices, bad)
        self.knn_indices = index_map[self.knn_indices]
        assert np.all(self.knn_indices[mask_2d] == -1)
        self.knn_dists[mask_2d] = np.inf

        assert self.knn_indices.shape == self.knn_dists.shape
        return copy.deepcopy(self.knn_indices), copy.deepcopy(self.knn_dists)
    
    def final(self):
        assert len(self.valid_original_indexes) == self.knn_indices.shape[0] == self.knn_dists.shape[0]
        mask = np.zeros(self.original_size, dtype=bool)
        mask[self.valid_original_indexes] = True
        return mask

### MNN METHODS ################################################################
### source: https://umap-learn.readthedocs.io/en/latest/mutual_nn_umap.html ####

def create_knn_matrix(knn_indices, knn_dists):
    assert knn_indices.shape == knn_dists.shape
    assert knn_indices.dtype == np.int32 and knn_dists.dtype == np.float64
    assert np.max(knn_indices) < len(knn_indices)
    assert np.array_equal(knn_indices[:,0], np.arange(len(knn_indices)))
    assert np.all(knn_dists[:,1:] > 0) and not np.any(np.isnan(knn_dists))
    
    rows = np.repeat(knn_indices[:,0], knn_indices.shape[1]-1)
    cols = knn_indices[:,1:].ravel()
    vals = knn_dists[:,1:].ravel()
    
    # remove missing elements before constructing matrix
    remove = (cols < 0) | (vals <= 0) | ~np.isfinite(cols) | ~np.isfinite(vals)
    rows = rows[~remove]
    cols = cols[~remove]
    vals = vals[~remove]
    if np.sum(remove) > 0:
        print(f"{np.sum(remove)} values removed during matrix construction")
    
    knn_matrix = sp.csr_matrix((vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))
    
    return knn_matrix    

# Prune non-reciprocated edges from matrix
def create_knn_from_matrix(knn_matrix):
    # Create mutual graph
    print(f"Creating KNN indices and dists...")
    lil = knn_matrix.tolil()
    lil.setdiag(0)
    # del knn_matrix, m
    del knn_matrix
    
    # Write output
    print("Writing output...")
    nrows = lil.shape[0]
    ncols = max(len(row) for row in lil.rows) + 1
    
    knn_indices = np.full((nrows, ncols), -1, dtype=np.int32)
    knn_dists = np.full((nrows, ncols), np.inf, dtype=np.float64)
    for i in range(nrows):
        cols = np.array(lil.rows[i]+[i], dtype=np.int32)
        vals = np.array(lil.data[i]+[0], dtype=np.float64)

        sorted_indices = np.argsort(vals)
        cols = cols[sorted_indices]
        vals = vals[sorted_indices]

        knn_indices[i, :len(cols)] = cols
        knn_dists[i, :len(vals)] = vals
    
    print("done")
    return knn_indices, knn_dists

# Prune non-reciprocated edges from matrix
def create_mnn_from_matrix(knn_matrix):
    # Create mutual graph
    print(f"Creating mutual graph...")
    m = np.abs(knn_matrix - knn_matrix.T) > np.min(knn_matrix.data)/2
    row, col = knn_matrix.nonzero()
    m.indptr = m.indptr.astype(np.int64)
    m.indices = m.indices.astype(np.int64)
    knn_matrix.data *= ~m[knn_matrix.astype(bool)].A1
    knn_matrix.eliminate_zeros()
    lil = knn_matrix.tolil()
    lil.setdiag(0)
    # del knn_matrix, m
    del knn_matrix
    
    # Write output
    print("Writing output...")
    nrows = lil.shape[0]
    ncols = max(len(row) for row in lil.rows) + 1
    
    mnn_indices = np.full((nrows, ncols), -1, dtype=np.int32)
    mnn_dists = np.full((nrows, ncols), np.inf, dtype=np.float64)
    for i in range(nrows):
        cols = np.array(lil.rows[i]+[i], dtype=np.int32)
        vals = np.array(lil.data[i]+[0], dtype=np.float64)

        sorted_indices = np.argsort(vals)
        cols = cols[sorted_indices]
        vals = vals[sorted_indices]

        mnn_indices[i, :len(cols)] = cols
        mnn_dists[i, :len(vals)] = vals
    
    print("done")
    return mnn_indices, mnn_dists

# Prune non-reciprocated edges
def create_mnn(knn_indices, knn_dists):
    assert knn_indices.shape == knn_dists.shape

    # Create mutual graph
    print(f"Creating mutual graph...")
    knn_matrix = create_knn_matrix(knn_indices, knn_dists).tocsr()
    m = np.abs(knn_matrix - knn_matrix.T) > np.min(knn_matrix.data)/2
    knn_matrix.data *= ~m[knn_matrix.astype(bool)].A1
    knn_matrix.eliminate_zeros()
    lil = knn_matrix.tolil()
    lil.setdiag(0)
    del knn_matrix, m
    
    # Write output
    print("Writing output...")
    nrows = lil.shape[0]
    ncols = max(len(row) for row in lil.rows) + 1
    
    mnn_indices = np.full((nrows, ncols), -1, dtype=np.int32)
    mnn_dists = np.full((nrows, ncols), np.inf, dtype=np.float64)
    for i in range(nrows):
        cols = np.array(lil.rows[i]+[i], dtype=np.int32)
        vals = np.array(lil.data[i]+[0], dtype=np.float64)

        sorted_indices = np.argsort(vals)
        cols = cols[sorted_indices]
        vals = vals[sorted_indices]

        mnn_indices[i, :len(cols)] = cols
        mnn_dists[i, :len(vals)] = vals
    
    print("done")
    return mnn_indices, mnn_dists

# Search to find path neighbors (todo: stop at dist=0)
def find_new_nn(indices, dists, out_neighbors, i_range):
    mnn_dists = [] 
    mnn_indices = []
    
    for i in i_range:
        min_indices = []
        min_distances = []
        
        heap = [(0,i)] ; heapq.heapify(heap) 
        mapping = {}
        seen = set()
        
        while len(min_distances) < out_neighbors and len(heap) > 0:
            dist, nn = heapq.heappop(heap)
            if nn in seen or nn < 0:
                continue
        
            min_indices.append(nn)
            min_distances.append(dist)
            seen.add(nn)
            
            for nn_i, nn_d in zip(indices[nn], dists[nn]):
                if nn_i in seen or nn_d <= 0:
                    continue
                distance = dist + nn_d
                if nn_i not in mapping or distance < mapping[nn_i]:
                    mapping[nn_i] = distance
                    heapq.heappush(heap, (distance, nn_i))
            
        if len(min_distances) < out_neighbors:
            for i in range(out_neighbors-len(min_distances)):
                min_indices.append(-1)
                min_distances.append(np.inf)
        
        mnn_indices.append(min_indices)
        mnn_dists.append(min_distances)
        
    return np.array(mnn_indices, dtype=np.int32), np.array(mnn_dists)

def find_path_neighbors(knn_indices, knn_dists, out_neighbors, n_jobs=-1):
    print("Finding new path neighbors...")
    assert np.all(np.sum(knn_indices[:,1:]>=0, axis=1) > 0), "ERROR: Some beads have no edges"
    
    from multiprocessing import Pool
    if n_jobs < 1:
        n_jobs = len(os.sched_getaffinity(0))

    ranges = np.array_split(range(len(knn_indices)), n_jobs)
    with Pool(processes=n_jobs) as pool:
        results = pool.starmap(find_new_nn, [(knn_indices, knn_dists, out_neighbors, i_range) for i_range in ranges])

    mnn_indices, mnn_dists = zip(*results)
    mnn_indices = np.vstack(mnn_indices)
    mnn_dists = np.vstack(mnn_dists)
    
    assert mnn_indices.shape == mnn_dists.shape
    assert mnn_indices.shape[0] == knn_indices.shape[0]
    assert np.max(mnn_indices) < len(mnn_indices)
    assert np.all(mnn_indices >= 0) and np.all(mnn_dists >= 0)
    assert np.array_equal(mnn_indices[:,0], np.arange(len(mnn_indices)))
    assert mnn_indices.dtype == np.int32 and mnn_dists.dtype == np.float64
    return mnn_indices, mnn_dists


### INITIALIZATION METHODS ################################################################
def leiden_init(knn_indices, knn_dists, n_neighbors, resolution_parameter = 160):
    # # Create graph edges with distances
    n_points, n_neighbors = knn_indices.shape
    row_indices = np.repeat(np.arange(n_points), n_neighbors - 1)  
    col_indices = knn_indices[:, 1:].ravel() 

    # Filter out edges where row or column indices are -1
    valid_edges_mask = (row_indices != -1) & (col_indices != -1)
    row_indices = row_indices[valid_edges_mask]
    col_indices = col_indices[valid_edges_mask]
    edges = np.column_stack((row_indices, col_indices))
    
    g = igp.Graph(edges=edges, directed=True)

    # run leidenalg
    resolution_parameter = resolution_parameter  
    partition_type = la.RBConfigurationVertexPartition  
    
    partition = la.find_partition(
        g,
        partition_type,
        resolution_parameter=resolution_parameter,
    )

    mem = np.array(partition.membership)
    num_clusters = len(np.unique(mem))
    
    print(f'number of clusters: {num_clusters}')
    print(f'modularity: {partition.modularity}')
    
    
    def inter_cluster_edge_calc(num_clusters, mem, g, weighted=True):
        inter_cluster_edges = np.zeros((num_clusters, num_clusters))
    
        edges = np.array([(mem[edge.source], mem[edge.target]) for edge in g.es])
        
        inter_cluster_edges_list = edges[edges[:, 0] != edges[:, 1]]
    
        counts = Counter(map(tuple, inter_cluster_edges_list))
        
        for (source_cluster, target_cluster), count in counts.items():
            inter_cluster_edges[source_cluster, target_cluster] += count
        
        return inter_cluster_edges

    ic_edges = inter_cluster_edge_calc(num_clusters, mem, g, weighted=False)
    ic_edges = ic_edges + ic_edges.T

    knn_indices, knn_dists = knn_descent(ic_edges, 150, metric="cosine", n_jobs=-1)
    knn_indices = knn_indices[:, :n_neighbors]
    knn_dists = knn_dists[:, :n_neighbors]
    knn = (knn_indices, knn_dists)
    
    mem_embeddings = my_umap(ic_edges, knn = knn, n_epochs = 20000, min_dist = 0.1, init = "spectral")
    mem_embeddings[:, 0] -= np.mean(mem_embeddings[:, 0])
    mem_embeddings[:, 1] -= np.mean(mem_embeddings[:, 1])
    
    fig, ax = plt.subplots()
    ax.scatter(mem_embeddings[:, 0], mem_embeddings[:, 1], s=1)
    ax.set_title('Leiden Initialization')
    
    init = mem_embeddings[mem]
    
    return init, ic_edges, fig, ax

### UMAP METHODS ################################################################
def my_umap(mat, knn, n_epochs, n_neighbors = 45, min_dist = 0.1, init="spectral"):
    reducer = UMAP(n_components = 2,
                   metric = "cosine",
                   spread = 1.0,
                   random_state = None,
                   verbose = True,
                   
                   precomputed_knn = knn if not (knn is None) else (None, None, None),
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   n_epochs = n_epochs,
                   init = init
                  )
    embedding = reducer.fit_transform(np.log1p(mat))
    return(embedding)
