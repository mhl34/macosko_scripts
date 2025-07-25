library(NMF)
library(qs)
library(Seurat)
library(furrr)
library(data.table)
library(magrittr)
library(SPARK)
library(RcppML)
library(ggplot2)
library(dplyr)
library(viridis)
library(scCustomize)
library(cowplot)
library(ks)
library(hexbin)
library(rnndescent)
library(patchwork)
library(qpdf)
library(entropy)
library(recall)
library(stats)
library(harmony)
library(glue) ; g=glue ; len=length
library(stringr)
library(zellkonverter)

# function: add spatial embeddings from metadata (if formatted as x_um and y_um)
# input: obj
# output: obj
add_spatial <- function(obj) {
  emb = obj@meta.data[,c("x_um","y_um")] ; colnames(emb) = c("f_1","f_2")
  obj[["spatial"]] <- CreateDimReducObject(embeddings = as.matrix(emb), key = "f_")
  return(obj)
}

# function: add spatial embeddings from metadata (if formatted as x_um and y_um)
# input: obj
# output: obj
add_spatial_tfmd <- function(obj) {
  emb = obj@meta.data[,c("x_um2","y_um2")] ; colnames(emb) = c("f_1","f_2")
  obj[["spatial_tfmd"]] <- CreateDimReducObject(embeddings = as.matrix(emb), key = "f_")
  return(obj)
}

add_dbscan = function(obj) {
  emb = obj@meta.data[,c("x","y")] ; colnames(emb) = c("d_1","d_2")
  obj[["spatial"]] <- CreateDimReducObject(embeddings = as.matrix(emb), key = "d_")
  return(obj)
}

ic_wm_finder = function(obj, gm_zones, wm_zones) {
  print('subset...')
  if (!('spatial' %in% names(obj@reductions))) {
    obj %<>% add_dbscan
  }
  obj = obj[,!is.na(obj$x)]
  gm = obj[,obj@meta.data[['zones_WM_C']] %in% gm_zones]
  
  print('chull...')
  gm_embed = as.data.frame(Embeddings(gm, reduction = 'spatial'))
  chull_res = chull(gm_embed)
  gm_points = gm_embed[chull_res,]
  
  print('mask...')
  obj_embed = as.data.frame(Embeddings(obj, reduction = 'spatial'))
  obj_embed[['x']] = obj_embed[['d_1']]
  obj_embed[['y']] = obj_embed[['d_2']]
  obj_embed = obj_embed[,c('x', 'y')]
  gm_points[['x']] = gm_points[['d_1']]
  gm_points[['y']] = gm_points[['d_2']]
  gm_points = gm_points[,c('x','y')]
  mask <- in.bounds(obj_embed, gm_points)
  wm_mask <- obj@meta.data[['zones_WM_C']] == 'WM'
  
  obj@meta.data[['zones_striatal']]= ifelse(mask & wm_mask, 'WM', 'Striatal')
  return(obj)
}

# function: un KNN neighbor finding (optimized parameters from M. Shabet)
# input: mat (matrix of loess smoothed), k (number of neighbors)
# output: return high quality knn for however many neighbors
find_knn <- function(mat, k, metric = "cosine", n_threads = 0) {
  matrix_knn <- rnnd_knn(data=mat,
                         k = k,
                         metric = metric,
                         n_trees=64,
                         low_memory=T,
                         max_candidates=60,
                         max_tree_depth=999999,
                         n_iters=512,
                         delta=0.0001,
                         n_threads = n_threads,
  )
  return(matrix_knn)
}

# function: un SPARK-X and return x genes
# input: obj (Seurat object), 
# optional: num_genes (number of genes to output, all if -1), reduction (name of dim reduction where spatial coordinates are stored)
# output: df (spark-x produced output)
run_sparkx <- function(obj, num_genes=-1, reduction='spatial', ncores = 16) {
  obj_counts = as.matrix(obj@assays$RNA$counts)
  obj_spatial_df = as.data.frame(Embeddings(obj, reduction = reduction))
  
  sparkx_res <- sparkx(obj_counts, obj_spatial_df, numCores=ncores, option="mixture")
  sparkx_res_sorted <- sparkx_res$res_mtest[order(sparkx_res$res_mtest$adjustedPval),]
  
  if (num_genes==-1) {
    return(sparkx_res_sorted)
  }
  return(sparkx_res_sorted, num_genes)
}

FindSpatiallyVariableFeatures <- function(obj, num_genes = 2000, pval_thresh = 1e-10, select_threshold = 100, select_flag = 'adjustedPval') {
  availCores = parallelly::availableCores()
  print('Get spatial information...')
  spatial_df = as.data.frame(Embeddings(obj, reduction = 'spatial'))
  print('Get counts and normalize...')
  obj_data = as.matrix(obj@assays$RNA$counts)
  obj_data_cell_norm = sweep(obj_data, 2, colSums(obj_data), "/")
  scaled_spatial_df = as.data.frame(scale(spatial_df))
  print('Run SPARK-X')
  obj_sparkx_res <- sparkx(obj_data, scaled_spatial_df, numCores=availCores, option="mixture")
  obj_sparkx_res_sort <- obj_sparkx_res$res_mtest[order(obj_sparkx_res$res_mtest[[select_flag]]),]
  num_select = min(sum(obj_sparkx_res_sort[[select_flag]] < pval_thresh), num_genes)
  VariableFeatures(obj) = rownames(obj_sparkx_res_sort[1:num_select,])
  print(paste0('P-Value Threshold: ', pval_thresh))
  print(paste0("Selected ", num_select, " spatially variable genes"))
  stopifnot(num_select >= select_threshold)
  return(obj)
}

# function: find ratio of the number of cells in a hexbin compared to the number of cells of a certain type
# input: obj, label, label_type
# output: df
find_relative_density <- function(obj, label_type = 'mmc_group') {
  coords <- Embeddings(obj, "spatial")
  celltypes <- obj[[label_type]]
  
  # coords: n×2 matrix of (x,y); celltypes: factor of length n
  coords_mat <- as.matrix(coords)
  types      <- unique(celltypes)[[1]]
  
  # choose a common bandwidth (h) or let ks estimate it
  H <- Hpi(coords_mat)  
  
  # overall density at each point
  f_all <- kde(x = coords_mat, H = H)
  dens_all_at_pts <- predict(f_all, x = coords_mat)
  
  # per-type normalized density
  norm_dens <- numeric(length(celltypes))
  for (t in types) {
    w <- as.numeric(celltypes == t)
    f_t <- kde(x = coords_mat, w = w, H = H)
    dens_t_at_pts <- predict(f_t, x = coords_mat)
    # ratio λ_t / λ_all
    norm_dens[celltypes == t] <- dens_t_at_pts[celltypes == t] / dens_all_at_pts[celltypes == t]
  }
  
  # add back to Seurat metadata
  obj$norm_dens <- norm_dens
  return(obj)
}

find_density = function(obj) {
  coords <- Embeddings(obj, "spatial")
  
  # coords: n×2 matrix of (x,y); celltypes: factor of length n
  coords_mat <- as.matrix(coords)
  
  # choose a common bandwidth (h) or let ks estimate it
  H <- Hpi(coords_mat)  
  
  # overall density at each point
  f_all <- kde(x = coords_mat, H = H)
  dens_all_at_pts <- predict(f_all, x = coords_mat)
  
  # add back to Seurat metadata
  obj$dens <- dens_all_at_pts
  return(obj)
}
