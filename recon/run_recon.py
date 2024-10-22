import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from collections import Counter
from helpers import *
from scipy.sparse import coo_matrix, csr_matrix
import scipy
import os
from umap import UMAP
import time
import igraph as igp
import leidenalg as la
from tqdm import tqdm

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
    # z_high = 3
    # sb1 = df.groupby(f'sb1_index').agg(umi=('umi', 'sum'), connections=('umi', 'size'), max=('umi', 'max')).reset_index()
    # hist_z(axes[0,0], np.log10(sb1['connections']), z_high)
    # axes[0,0].set_xlabel('Connections')
    # axes[0,0].set_title('sb1 connections')
    # hist_z(axes[0,1], np.log10(sb1['max']))
    # axes[0,1].set_xlabel('Max')
    # axes[0,1].set_title('sb1 max')

    # logcon = np.log10(sb1['connections'])
    # high = np.where(logcon >= np.mean(logcon) + np.std(logcon) * z_high)[0]
    # noise = np.where(sb1['max'] <= 5)[0]
    # sb1_remove = reduce(np.union1d, [high, noise])
    # df = df[~df['sb1_index'].isin(sb1_remove)]
    
    # meta["sb1_high"] = len(high)
    # meta["sb1_removed"] = len(sb1_remove)
    # meta["umi_half"] = sum(df["umi"])
    # print(f"{len(high)} high R1 beads ({len(high)/len(sb1)*100:.2f}%)")
    # diff = meta['umi_init']-meta['umi_half'] ; print(f"{diff} R1 UMIs filtered ({diff/meta['umi_init']*100:.2f}%)")

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
    noise = np.where(sb2['max'] <= 5)[0]
    sb2_remove = reduce(np.union1d, [high, low, noise])
    # sb2_remove = reduce(np.union1d, [low, noise])
    # sb2_remove = reduce(np.union1d, [high, low])
    df = df[~df['sb2_index'].isin(sb2_remove)]
    
    meta["sb2_high"] = len(high)
    meta["sb2_low"] = len(low)
    meta["sb2_noise"] = len(noise)
    meta["sb2_removed"] = len(sb2_remove)
    meta["umi_final"] = sum(df["umi"])
    print(f"{len(high)} high R2 beads ({len(high)/len(sb2)*100:.2f}%)")
    print(f"{len(low)} low R2 beads ({len(low)/len(sb2)*100:.2f}%)")
    print(f"{len(noise)} noise R2 beads ({len(noise)/len(sb2)*100:.2f}%)")
    # diff = meta['umi_half']-meta['umi_final'] ; print(f"{diff} R2 UMIs filtered ({diff/meta['umi_half']*100:.2f}%)")
    
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

def my_umap(mat, n_epochs, init=init, metric="cosine", repulsion_strength = 1, learning_rate = 1, precompute = False):
    reducer = UMAP(n_components = 2,
                   metric = metric,
                   spread = 1.0,
                   random_state = None,
                   learning_rate = learning_rate,
                   repulsion_strength = 1,
                   verbose = True,
                   precomputed_knn = (knn_indices, knn_dists) if precompute else (None, None, None),
                   n_neighbors = n_neighbors,
                   min_dist = min_dist,
                   n_epochs = n_epochs,
                   init = init,
                   n_jobs = -1
                  )
    embedding = reducer.fit_transform(np.log1p(mat))
    return(embedding)

parser = argparse.ArgumentParser()
parser.add_argument("-md", "--min_dist", dest="min_dist", default = 0.1, help="minimum distance for umap")
parser.add_argument("-nn", "--n_neighbors", dest="n_neighbors", default = 45, help="number of neighbors")
parser.add_argument("-e", "--epochs", default=1000, dest="n_epochs", help="input the number of epochs to run")
parser.add_argument("-i", "--in_dir", default=".", dest="in_dir", help="input directory")
parser.add_argument("-o", "--out_dir", default=".", dest="out_dir", help="output directory")
args = parser.parse_args()

min_dist = args.min_dist
n_neighbors = args.n_neighbors
n_epochs = args.n_epochs
in_dir = args.in_dir
out_dir = args.out_dir

print(f'min dist: {min_dist}')
print(f'n neighbors: {n_neighbors}')
print(f'epochs: {n_epochs}')

print("\nReading the matrix...")
df = pd.read_csv(f'{in_dir}/matrix.csv.gz', compression='gzip', quotechar='"') 
df = df[['sb1_index', 'sb2_index', 'umi']]
df = df[df['umi'] != 1]
df.sb1_index -= 1 # convert from 1- to 0-indexed
df.sb2_index -= 1 # convert from 1- to 0-indexed

print("\nFiltering the beads...")
df, uniques1, uniques2, fig, meta = connection_filter(df)

# Rows are the beads you wish to recon
# Columns are the features used for judging similarity
mat = coo_matrix((df['umi'], (df['sb1_index'], df['sb2_index']))).tocsr()
scipy.sparse.save_npz(os.path.join(in_dir, 'mat_unified.npz'), mat)
del df

### Compute the KNN ############################################################

print("\nComputing the KNN...")
n_neighbors_max = 150
knn_indices, knn_dists = knn_descent(np.log1p(mat), n_neighbors_max)
knn_indices[:, 0] = np.arange(knn_indices.shape[0])
np.savez_compressed(os.path.join(out_dir, "knn_unified.npz"), indices=knn_indices, dists=knn_dists)

print('\nRun MNN...')
mnn_indices, mnn_dists = create_mnn(knn_indices, knn_dists)

# bad beads
print('\nRemove bad beads...')
bad_mask = (mnn_indices[:, 1] == -1) | (mnn_dists[:, 1] > 0.95)
bad_bead_indices = np.where(bad_mask)[0]
good_beads_indices = np.where(~bad_mask)[0]

mat = mat[good_beads_indices]
mnn_mask = KNNMask(mnn_indices, mnn_dists)
mnn_indices, mnn_dists = mnn_mask.remove(bad_bead_indices)

print('\nFind path neighbors...')
mnn_indices2, mnn_dists2 = find_path_neighbors(mnn_indices, mnn_dists, n_neighbors, n_jobs=-1)

# bad beads
print('\nRemove bad beads...')
bad_mask = (mnn_indices2[:, n_neighbors-1] == -1) | (mnn_dists2[:, 1] > 0.95)
if bad_mask.sum() > 0:
    bad_bead_indices = np.where(bad_mask)[0]
    good_beads_indices = np.where(~bad_mask)[0]
    
    mat = mat[good_beads_indices]
    mnn_mask = KNNMask(mnn_indices2, mnn_dists2)
    mnn_indices2, mnn_dists2 = mnn_mask.remove(bad_bead_indices)
    
knn_indices = mnn_indices2[:, :n_neighbors]
knn_dists = mnn_dists2[:, :n_neighbors]

scipy.sparse.save_npz(f'{out_dir}/mat_mask.npz', mat)
np.savez(f'{out_dir}/mnn.npz', indices = knn_indices, dists = knn_dists)

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

mem = np.array(partition.membership)
np.savez(f'{out_dir}/membership.npz', membership = mem))

print("Leiden partition complete")

print(f'number of clusters: {len(np.unique(partition.membership))}')
print(f'modularity: {partition.modularity}')
print(f'Leiden clustering: {end - start} seconds')

print("Find inter_cluster_edges")
ic_edges = inter_cluster_edge_calc(partition, g, weighted = False)
ic_edges = ic_edges + ic_edges.T

np.savez(f'{out_dir}/ic_edges.npz', ic_edges = ic_edges)

init = 'spectral'

start = time.time()
mem_embeddings = my_umap(ic_edges, n_epochs = 2000, metric = 'cosine', precompute = False)
mem_embeddings[:, 0] -= np.mean(mem_embeddings[:, 0])
mem_embeddings[:, 1] -= np.mean(mem_embeddings[:, 1])
end = time.time()

plt.scatter(mem_embeddings[:, 0], mem_embeddings[:, 1], s = 1)
plt.title(f'New SB1 selection')
plt.savefig(f'{out_dir}/mem_embeddings.png')
np.savez(f'{out_dir}/mem_embeddings.npz', embeddings = mem_embeddings)

init = mem_embeddings[mem]
print(init.shape)

print('run umap')
embeddings = my_umap(mat, n_epochs, metric = 'cosine', precompute = True)

print('save embeddings')
np.savez(f'{out_dir}embeddings.npz', embeddings = embeddings)
