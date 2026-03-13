import os
import scanpy as sc
from label_transfer_helpers import *
from pathlib import Path
import numpy as np
import pandas as pd
import pandas.api.types as pdt
import matplotlib.pyplot as plt
from scipy import sparse
import anndata as ad
from scipy.spatial.distance import cdist

# input: data, groupby (which columns to group by), weighted_cols (dict like {"pct_intronic": "nCount_RNA", "pct_mt": "nCount_RNA"}), agg (mean or sum)
# output: pseudobulked object
def pseudobulk_anndata(adata, groupby, weighted_cols=None, agg="sum"):
    if isinstance(groupby, str):
        groupby = [groupby]
    metadata = adata.obs.copy()
    expr = adata.X
    groups = metadata.groupby(groupby, observed=True).indices
    pseudobulk_expr = []
    pseudobulk_meta = []
    for group_key, cell_idx in groups.items():
        # expression slice
        X = expr[cell_idx]

        # aggregate expression
        if sparse.issparse(X):
            expr_sum = np.array(X.sum(axis=0)).flatten()
            if agg == "mean":
                expr_sum = expr_sum / X.shape[0]
        else:
            expr_sum = X.sum(axis=0)
            if agg == "mean":
                expr_sum = expr_sum / X.shape[0]
        pseudobulk_expr.append(expr_sum)
        group_meta = metadata.iloc[cell_idx]
        agg_meta = {}
        for col in group_meta.columns:
            if weighted_cols and col in weighted_cols:
                weight_col = weighted_cols[col]
                weights = group_meta[weight_col].to_numpy()
                values = group_meta[col].to_numpy()
                agg_meta[col] = np.average(values, weights=weights)
            elif pdt.is_numeric_dtype(group_meta[col]):
                agg_meta[col] = group_meta[col].mean()
            else:
                if pdt.is_categorical_dtype(group_meta[col]) or pdt.is_object_dtype(group_meta[col]):
                    # use mode if available, else fallback to first value
                    mode_val = group_meta[col].mode()
                    agg_meta[col] = mode_val.iloc[0] if not mode_val.empty else group_meta[col].iloc[0]
                else:
                    agg_meta[col] = group_meta[col].iloc[0]
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for k, v in zip(groupby, group_key):
            agg_meta[k] = v
        pseudobulk_meta.append(agg_meta)
    pseudobulk_expr = np.vstack(pseudobulk_expr)
    pseudobulk_meta = pd.DataFrame(pseudobulk_meta)
    pseudobulk_meta.index = [
        "-".join(map(str, row)) if isinstance(row, tuple) else str(row)
        for row in groups.keys()
    ]
    pb_adata = ad.AnnData(
        X=sparse.csr_matrix(pseudobulk_expr) if sparse.issparse(expr) else pseudobulk_expr,
        obs=pseudobulk_meta,
        var=adata.var.copy()
    )
    return pb_adata


def process(adata, batch_key = 'donor_id', harmony_key = None, regress_key = None, n_top_genes = 2000, n_jobs = 16):
    print('normalize and log...')
    if "log1p" not in adata.uns:
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    else:
        print('already normalized and logged')
    print('highly variable genes...')
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    print('scale counts...')
    def scale_by_batch(adata: ad.AnnData, batch_key: str) -> ad.AnnData:
        return ad.concat(
        {
            k: sc.pp.scale(adata[idx], copy=True, zero_center=False)
            for k, idx in adata.obs.groupby(batch_key).indices.items()
         },
       merge="first"
    )
    adata = scale_by_batch(adata,batch_key)
    if regress_key != None:
        print('regress out...')
        sc.pp.regress_out(adata, regress_key, n_jobs = n_jobs)
    print('pca...')
    sc.tl.pca(adata)
    if harmony_key != None:
        print('run harmony...')
        sc.external.pp.harmony_integrate(adata, harmony_key)
    print('find neighbors...')
    sc.pp.neighbors(adata)
    print('cluster...')
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, key_added="leiden")
    print('run umap...')
    sc.tl.umap(adata)
    return adata


def hierarchical_merging(adata, groupby, embedding_key="X_harmony"):
    """
    Iteratively merges groups based on centroid distances of embeddings,
    saving each step's grouping in adata.obs["merge_step_{i}"].
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with Harmony embeddings in .obsm[embedding_key].
    groupby : str
        Column in adata.obs to group by (initial clusters).
    embedding_key : str, optional
        Key in adata.obsm containing the Harmony embeddings (default "X_harmony").
    """
    # Start with initial grouping
    labels = adata.obs[groupby].astype(str).copy()
    adata.obs[f"merge_step_{labels.nunique()}"] = labels

    # Run iterative merging until one group remains
    step = labels.nunique()
    while step > 1:
        # Compute centroids for current groups
        centroids = (
            pd.DataFrame(adata.obsm[embedding_key])
            .groupby(labels.values)
            .mean()
        )

        # Compute pairwise distances
        dists = cdist(centroids.values, centroids.values, metric="euclidean")
        np.fill_diagonal(dists, np.inf)  # avoid self-merging

        # Find closest pair
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        group_i, group_j = centroids.index[i], centroids.index[j]

        # Merge: assign group_j cells to group_i
        new_labels = labels.copy()
        new_labels[new_labels == group_j] = group_i

        # Save updated labels
        step -= 1
        adata.obs[f"merge_step_{step}"] = new_labels
        labels = new_labels

    return adata
