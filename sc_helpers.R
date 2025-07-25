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
library(enrichR)
library(cowplot)
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
library(igraph)
library(data.table)
library(leidenbase)

# function: cleans object to < 5% mito and placed
# input: obj
# output: obj
# cleaning <- function(obj) {
#   obj %>% subset(., percent.mt < 5 & pct.intronic > 0.60)
# }

# function: processes object with a custom gene_set (with harmony)
# input: obj
# output: obj
process_no_dimred <- function(obj, vg_set, res=0.8, n.epochs=NULL, var = 'library', metric = 'euclidean', regress_var = NULL) {
  vg_set = vg_set[vg_set %in% rownames(obj)]
  VariableFeatures(obj) = vg_set
  obj <- obj %>%
    Seurat::SCTransform(
      assay            = "RNA",
      new.assay.name   = "SCT",
      residual.features = vg_set,
      vars.to.regress  = regress_var,    # e.g. also regress out mito %
      verbose          = T
    ) %>% 
    Seurat::RunPCA(assay = "SCT", verbose=F) %>%
    RunHarmony(group.by.vars = var) %>% 
    Seurat::FindNeighbors(assay = 'SCT', features = vg_set, prune.SNN = 0, n.trees = 100, annoy.metric = metric) %>%
    Seurat::FindClusters(graph.name = 'SCT_snn', n.start = 20, n.iter = 20, algorithm = 4, method = 'igraph', resolution = res) %>%
    Seurat::RunUMAP(dims = 1:30, verbose=F, n.epochs=n.epochs, metric = metric)
}

# function: processes object with a custom gene_set (with harmony) (old)
# input: obj
# output: obj
process_no_dimred_old <- function(obj, vg_set, res=0.8, n.epochs=NULL, var = 'library', metric = 'cosine', regress_var = NULL) {
  vg_set = vg_set[vg_set %in% rownames(obj)]
  VariableFeatures(obj) = vg_set
  obj <- obj %>%
    Seurat::ScaleData(vars.to.regress = regress_var) %>%
    Seurat::RunPCA(verbose=F) %>%
    RunHarmony(group.by.vars = var) %>% 
    Seurat::FindNeighbors(assay = 'harmony', features = vg_set, prune.SNN = 0, n.trees = 100, annoy.metric = metric) %>%
    Seurat::FindClusters(n.start = 20, n.iter = 20, algorithm = 4, resolution = res) %>%
    Seurat::RunUMAP(dims = 1:30, verbose=F, n.epochs=n.epochs, metric = metric)
}

# function: processes object (with harmony)
# input: obj
# output: obj
process <- function(obj, res=0.8, n.epochs=NULL, var = 'library', vars.to.regress = NULL) {
  obj <- obj %>%
    Seurat::NormalizeData() %>%
    Seurat::FindVariableFeatures() %>%
    Seurat::ScaleData(vars.to.regress = vars.to.regress) %>%
    Seurat::RunPCA(verbose=F) %>%
    RunHarmony(group.by.vars = var) %>% 
    Seurat::FindNeighbors(dims=1:30, n.trees = 100, reduction = 'harmony') %>%
    Seurat::FindClusters(n.start = 20, n.iter = 20, algorithm = 4, resolution = res) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F, n.epochs=n.epochs, reduction = 'harmony')
}

process_sctransform <- function(obj,res=0.8,n.epochs=NULL,var="library",vars.to.regress=NULL){
  obj%>%
    Seurat::SCTransform(assay="RNA",new.assay.name="SCT",vars.to.regress=vars.to.regress,verbose=T)%>%
    Seurat::RunPCA(assay="SCT",verbose=FALSE)%>%
    RunHarmony(group.by.vars = var) %>% 
    Seurat::FindNeighbors(dims=1:30, n.trees = 100, reduction = 'harmony') %>%
    Seurat::FindClusters(n.start = 20, n.iter = 20, algorithm = 4, resolution = res) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F, n.epochs=n.epochs, reduction = 'harmony')
}

# function: processes object spatially (with harmony)
# input: obj
# output: obj
process_spatial <- function(obj, num_genes = 2000, res=0.8, n.epochs=NULL, var = 'library', pval_thresh = 1e-10, select_threshold = 100, select_flag = 'adjustedPval') {
  obj <- obj %>%
    Seurat::NormalizeData() %>%
    FindSpatiallyVariableFeatures(., num_genes = num_genes, pval_thresh = pval_thresh, select_threshold = select_threshold, select_flag = select_flag) %>%
    Seurat::ScaleData() %>%
    Seurat::RunPCA(verbose=F) %>%
    RunHarmony(group.by.vars = var) %>% 
    Seurat::FindNeighbors(dims=1:30, n.trees = 100, reduction = 'harmony') %>%
    Seurat::FindClusters(n.start = 20, n.iter = 20, algorithm = 4, resolution = res) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F, n.epochs=n.epochs, reduction = 'harmony')
}

# function: processes object spatially (with harmony)
# input: obj
# output: obj
process_custom_old <- function(obj, vg_set, res=0.8, n.epochs=NULL, var = 'library', metric = 'euclidean', regress_var = NULL) {
  vg_set = vg_set[vg_set %in% rownames(obj)]
  VariableFeatures(obj) = vg_set
  obj <- obj %>%
    Seurat::NormalizeData() %>%
    Seurat::ScaleData(vars.to.regress = regress_var) %>%
    Seurat::RunPCA(verbose=F) %>%
    RunHarmony(group.by.vars = var) %>%
    Seurat::FindNeighbors(dims=1:30, n.trees = 100, reduction = 'harmony', annoy.metric = metric) %>%
    Seurat::FindClusters(n.start = 20, n.iter = 20, algorithm = 4, method = 'igraph', resolution = res) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F, n.epochs=n.epochs, reduction = 'harmony', metric = metric)
}

# function: processes object with a custom gene_set (with harmony)
# input: obj
# output: obj
process_custom <- function(obj, vg_set, res=0.8, n.epochs=NULL, var = 'library', metric = 'euclidean', regress_var = NULL) {
  vg_set = vg_set[vg_set %in% rownames(obj)]
  VariableFeatures(obj) = vg_set
  obj <- obj %>%
    Seurat::SCTransform(
      assay            = "RNA",
      new.assay.name   = "SCT",
      residual.features = vg_set,
      vars.to.regress  = regress_var,    # e.g. also regress out mito %
      verbose          = T
    ) %>% 
    Seurat::RunPCA(assay = "SCT", verbose=F) %>%
    RunHarmony(group.by.vars = var, theta = rep(2, len(var))) %>%
    Seurat::FindNeighbors(dims=1:30, n.trees = 100, reduction = 'harmony', annoy.metric = metric) %>%
    Seurat::FindClusters(n.start = 20, n.iter = 20, algorithm = 4, method = 'igraph', resolution = res) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F, n.epochs=n.epochs, reduction = 'harmony', metric = metric)
}

# function: processes object (without harmony)
# input: obj
# output: obj
process_no_harmony <- function(obj, res=0.8, n.epochs=NULL) {
  obj <- obj %>%
    Seurat::NormalizeData() %>%
    Seurat::FindVariableFeatures() %>%
    Seurat::ScaleData() %>%
    Seurat::RunPCA(verbose=F) %>%
    Seurat::FindNeighbors(dims=1:30, n.trees = 100, reduction = 'pca') %>%
    Seurat::FindClusters(., n.start = 20, n.iter = 20, algorithm = 4, resolution = res) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F, n.epochs=n.epochs, reduction = 'pca')
}

process_spatial_no_harmony <- function(obj, num_genes = 2000, res=0.8, n.epochs=NULL, var = 'library', pval_thresh = 1e-10, select_threshold = 100, select_flag = 'adjustedPval') {
  obj <- obj %>%
    Seurat::NormalizeData() %>%
    FindSpatiallyVariableFeatures(., num_genes = num_genes, pval_thresh = pval_thresh, select_threshold = select_threshold, select_flag = select_flag) %>%
    Seurat::ScaleData() %>%
    Seurat::RunPCA(verbose=F) %>%
    Seurat::FindNeighbors(dims=1:30, n.trees = 100, reduction = 'pca') %>%
    Seurat::FindClusters(n.start = 20, n.iter = 20, algorithm = 4, resolution = res) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F, n.epochs=n.epochs, reduction = 'pca')
}

# function: run harmony
# input: obj
# output: obj
run_harmony <- function(obj, key) {
  obj <- RunHarmony(obj, key)
}

# function: iterative clustering regime (with depth 2)
# input: obj
# output: obj
iterative_clustering <- function(seurat_obj, resolution = 0.8, min_cells = 200) {
  seurat_obj %<>% process
  cluster_ids = levels(seurat_obj$recall_clusters)
  seurat_obj[['depth_1']] = seurat_obj$recall_clusters
  
  for (cluster in cluster_ids) {
    depth = 2
    sub_seurat_obj = subset(seurat_obj, recall_clusters == cluster)
    sub_seurat_obj %<>% process(., res = resolution)
    subcluster_cells = colnames(sub_seurat_obj)
    
    message(paste("Cluster ID:", cluster, "- Number of cells:", dim(sub_seurat_obj)[2]))
    if (length(subcluster_cells) < min_cells) {
      message(paste("Skipping subclustering for cluster", cluster, "- too few cells (", length(subcluster_cells), ")"))
      next
    }
    message(paste("Subclustering cluster:", cluster, "with", length(subcluster_cells), "cells"))
    
    col_name <- sprintf("depth_%d", depth)
    
    if (!(col_name %in% colnames(seurat_obj))) {
      seurat_obj@meta.data[, col_name] = NA
    }
    
    seurat_obj@meta.data[, col_name] <- as.character(seurat_obj@meta.data[, col_name])
    seurat_obj@meta.data[subcluster_cells, col_name] <- as.character(sub_seurat_obj[["seurat_clusters"]])
    gc()
  }
  return(seurat_obj)
}

# function: remove mitochondrial genes
# input: obj
# output: obj
remove_mito <- function(obj) {
  mt_genes <- grep("^MT-", rownames(obj), value = TRUE)
  obj_noMT <- subset(
    x        = obj,
    features = setdiff(rownames(obj), mt_genes)
  )
  return(obj_noMT)
}

# function: find neighborhoods FOR type 1 cells OF type 2 cells. if type 2 == NULL, then do against all.
# input: obj, type1, type2, n_neighbors
# input: return an index of cell barcodes that are neighbors (idx, dist, type)
knn_diff_types <- function(obj, type1 = NULL, type2 = NULL, n_neighbors = 45, type_label1 = 'mmc_group', type_label2 = 'mmc_group', embed_type = 'spatial') {
  print('Subset into different types...')
  if (is.null(type1)) {
    type1_obj = obj
  } else {
    type1_obj = obj[,obj@meta.data[[type_label1]] %in% type1]
  }
  if (is.null(type2)) {
    type2_obj = obj
  } else {
    type2_obj = obj[,obj@meta.data[[type_label2]] %in% type2]
  }
  
  print('Extract embeddings...')
  type1_embed = as.data.frame(Embeddings(type1_obj, reduction = embed_type))
  type2_embed = as.data.frame(Embeddings(type2_obj, reduction = embed_type))
  type1_barcodes = colnames(type1_obj)
  type2_barcodes = colnames(type2_obj)
  
  print('Run KNN...')
  index <- rnnd_build(type2_embed, k = n_neighbors)
  knn_res <- rnnd_query(
    index = index,
    query = type1_embed,
    k = n_neighbors
  )
  
  dims <- dim(knn_res$idx)
  result <- matrix(
    type2_obj@meta.data[[type_label2]][knn_res$idx],
    nrow = as.numeric(dims[1]),
    ncol = as.numeric(dims[2])
  )
  print('DONE')
  return(list(idx = knn_res$idx, dist = knn_res$dist, celltype_mat = result, barcodes1 = type1_barcodes, barcodes2 = type2_barcodes))
}


wm_gm_finder = function(obj, celltype, celltype_label = 'Subclass_name') {
  knn_res = knn_diff_types(obj, type1 = celltype, type2 = c('STR D1 MSN'), n_neighbors = 1, type_label1 = celltype_label, type_label2 = 'Subclass_name')
  gm_wm_distinction = ifelse(knn_res$dist[,1] < 100, 'GM', 'WM')
  return(gm_wm_distinction)
}

# get the counts of each neighboring 
get_celltypes_count = function(celltype_mat) {
  # Get all unique strings
  all_strings <- sort(unique(as.vector(celltype_mat)))
  
  # For each row, count occurrences
  count_mat <- t(apply(celltype_mat, 1, function(row) {
    table(factor(row, levels = all_strings))
  }))
  
  return(count_mat)
}

run_leiden = function(knn_idx, knn_dist) {
  n <- nrow(knn_idx)
  k <- ncol(knn_idx)
  
  # Build source and target vectors
  print('Build vectors...')
  src <- rep(1:n, each = k)
  tgt <- as.vector(knn_idx)
  wts <- exp(-as.vector(knn_dist))
  
  # Combine and deduplicate (keep (min, max) ordering)
  print('Add edges...')
  # Drop self-edges and build data.frame
  edge_dt <- data.table(
    from = pmin(src, tgt),
    to = pmax(src, tgt),
    weight = wts
  )
  
  # Check again
  stopifnot(ncol(edge_dt) >= 2)
  
  # Remove duplicates
  edge_dt <- unique(edge_dt)
  
  # Create graph
  print('Create graph...')
  g <- graph_from_data_frame(edge_dt, directed = FALSE)
  
  # Run Leiden
  print('Run Leiden...')
  clustering <- leiden_find_partition(g, partition_type = "RBConfiguration", resolution_parameter = 2, num_iter = 10, verbose = T)
  
  # Cluster membership
  print('Get memberships...')
  membership <- clustering$membership
  
  return(membership)
}

run_dropsift = function(obj) {
  return(NA)
}

go_enrichment = function(gene_list) {
  dbs = c("GO_Molecular_Function_2023", "GO_Cellular_Component_2023",
          "GO_Biological_Process_2023")
  enriched <- enrichr(gene_list, dbs)
  return(enriched)
}

run_cosg = function(obj, mean = 1, label = 'seurat_clusters', n_genes = 100) {
  Idents(obj) = as.vector(obj@meta.data[[label]])
  markers = cosg(
    obj,
    groups='all',
    assay='RNA',
    slot='data',
    mu=mean,
    n_genes_user=n_genes)
  
  return(list(names = markers$names[,order(colnames(markers$names))],
              scores = markers$scores[,order(colnames(markers$scores))]))
}

labeled_load = function(dir) {
  obj = qread(paste0(dir, 'merged_obj.qs'))
  df = read.csv(paste0(dir, 'mapping_output.csv'), comment.char = '#')
  obj %<>% mmc_metadata(., df)
  return(obj)
}

add_all_ucell = function(obj, markers_mat) {
  markers_list = as.list(markers_mat)
  obj %<>% AddModuleScore_UCell(., markers_list)
  return(obj)
}

cosg_threshold = function(markers, threshold = 0.01) {
  cosg_names = markers$names
  cosg_scores = markers$scores
  gene_lists = list()
  for (i in colnames(cosg_names)) {
    gene_lists[[i]] = cosg_names[,i][(cosg_scores[,i] >= threshold)]
  }
  return(gene_lists)
}
