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
library(SingleCellExperiment)
source('/broad/macosko/leematth/helpers/sc_helpers.R')
source('/broad/macosko/leematth/helpers/spatial_helpers.R')
source('/broad/macosko/leematth/helpers/anno_helpers.R')

remove_pseudogenes = function(obj) {
  mito_genes = substr(rownames(obj), 1, 3) == 'MT-'
  ribo_genes = substr(rownames(obj), 1, 3) %in% c("RPS", "RPL")
  obj = obj[!(mito_genes | ribo_genes),]
  return(obj)
}

celltype_subsetting = function(objs_list, type_label = 'STR D1 MSN', level = 'Subclass_name') {
  objs = list()
  for (obj_path in objs_list) {
    print(obj_path)
    obj = qread(obj_path)
    objs[[obj_path]] = obj[,obj@meta.data[level] == type_label]
    rm(obj)
    gc()
  }
  merged_obj = merge(x = objs[[1]], y = as.vector(objs[2:len(obj_paths)]), merge.data = TRUE)
  return(merged_obj)
}

sample_obj <- function(obj, num) {
  return(obj[, sample(seq_len(ncol(obj)), num, replace=FALSE)])
}

export_seurat_to_10x <- function(seurat_obj, outdir = "./exported") {
  if (!require(Matrix)) install.packages("Matrix")
  library(Matrix)
  
  # Create output directory
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  
  # Extract expression matrix (use `layer` instead of `slot`)
  mat <- GetAssayData(seurat_obj, assay = "RNA", layer = "counts")
  
  # Write Matrix Market file
  Matrix::writeMM(mat, file = file.path(outdir, "matrix.mtx"))
  
  # Write barcodes
  write.table(
    colnames(mat),
    file = file.path(outdir, "barcodes.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE
  )
  
  # Write genes/features
  features <- data.frame(
    gene_id = rownames(mat),
    gene_name = rownames(mat),
    stringsAsFactors = FALSE
  )
  write.table(
    features,
    file = file.path(outdir, "genes.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE
  )
}

merge_obj = function(in_dir, indexes) {
  print('Load objects...')
  objs = lapply(indexes, function(index) { 
    obj = qread(paste0(in_dir, index, '/seurat.qs'))
    obj$library = index
    obj$num_dbscan = obj@misc$coords$clusters
    return(obj) 
  })
  
  if (length(objs) < 2) {
    print('Less than 2 objects')
    return(NA)
  }
  
  print('Merge objects...')
  obj = merge(x = objs[[1]], y = c(objs[2:length(objs)]), merge.data = TRUE)
  if (len(Layers(obj[['RNA']])) > 3) {
    print('Join layers...')
    obj %<>% JoinLayers
  }
  stopifnot(length(indexes) == length(objs))
  for (i in seq_along(indexes)) {
    Misc(obj, indexes[[i]]) <- Misc(objs[[i]])
  }
  rm(objs)
  gc()
  
  print('Add spatial embeddings...')
  emb_obj = obj@meta.data[,c("x", "y")] ; colnames(emb_obj) = c("s_1","s_2")
  obj[["spatial"]] <- CreateDimReducObject(embeddings = as.matrix(emb_obj), key = "s_")
  
  print('Process...')
  obj %<>% process
  
  return (obj)
}

# function: load expression matrix from dge
# input: path
# output: object
load.expression.matrix <- function(path, use.skip = TRUE) {
  
  if (use.skip) {
    
    # used to skip header information
    dge <- fread(path, header = TRUE, skip = "ACGT")
    
  } else {
    
    dge <- fread(path, header = TRUE)
    
  }
  
  gene.names <- dge$GENE
  dge$GENE <- NULL
  dge <- as.matrix(dge)
  dge <- as(dge, "sparseMatrix")
  rownames(dge) <- gene.names
  
  return(dge)
}

# function: create bican object given date, library name, and the number of reactions
# input: date, library.name, num.rxn, reference (optional), base_dr (optional), cell.selection (optional)
# output: obj
create.bican.obj = function(date, library.name, num.rxn, reference.use = "GRCh38-2020-A.isa.exonic+intronic", base.dir = "/broad/bican_um1_mccarroll/RNAseq/data/libraries", cell.selection = "svm_nuclei_10X") {
  library.names <- paste0(library.name, 1:num.rxn)
  prefixes <- paste0("rxn", 1:num.rxn)
  
  n.libraries <- length(library.names)
  
  cell.selection.dirs <- paste0(base.dir, "/", date, "_", library.names, "/", reference.use, "/cell_selection/", cell.selection)
  std.analysis.dirs <- paste0(base.dir, "/", date, "_", library.names, "/", reference.use, "/std_analysis/", cell.selection)
  output.paths <- paste0(std.analysis.dirs, "/", library.names, ".", cell.selection)
  dge.paths <- paste0(std.analysis.dirs, "/", library.names, ".", cell.selection, ".donors.digital_expression.txt.gz")
  cell.features.paths <- paste0(cell.selection.dirs, "/", library.names, ".", cell.selection, ".cell_features.RDS")
  donor.assignment.paths <- paste0(std.analysis.dirs, "/", library.names, ".", cell.selection, ".donor_assignments.txt")
  # summary.data.paths <- paste0(std.analysis.dirs, "/scPred/summary.txt")
  
  objs = lapply(output.paths, function(path) { 
    dge.path = paste0(path, ".donors.digital_expression.txt.gz")
    dge = load.expression.matrix(dge.path)
    donor_assignment = read.csv(paste0(path, ".donor_cell_map.txt"), comment.char = '#', sep = '\t')
    obj = CreateSeuratObject(counts=dge)
    obj$library = str_split(path, '/')[[1]][7]
    obj$best_donor = donor_assignment$bestSample
    return(obj) 
  })
  
  obj = merge(x = objs[[1]], y = as.vector(objs[2:num.rxn]), merge.data = TRUE)
  obj %<>% JoinLayers
  
  obj$nCount_RNA = colSums(obj@assays$RNA$counts)
  obj$logumi = log10(obj$nCount_RNA)
  
  rm(objs)
  gc()
  
  return(obj)
}

# function convert seurat object to h5ad
# input: obj, output_dir
# output: NA (saves .h5ad to output_dir)
seurat_to_h5ad = function(obj, output_dir) {
  sce <- as.SingleCellExperiment(obj)
  writeH5AD(sce, output_dir)
}

# function convert bican object to seurat
# input: obj, output_dir
# output: NA (saves .h5ad to output_dir)
bican_to_seurat = function(obj_path) {
  sce <- readH5AD(obj_path)
  rd_subset <- as.data.frame(rowData(sce)[, c(
    "gene_symbol",
    "n_cells",
    "feature_is_filtered"
  )])
  rownames(sce) = rd_subset$gene_symbol
  seurat_obj <- as.Seurat(
    sce,
    counts = "X",   # use the “counts” assay for raw counts
    data   = NULL        # NULL means don’t override the default “data” slot
  )
  seurat_obj %<>% RenameAssays(., 'originalexp', 'RNA')
  return(seurat_obj)
}

# FROM MATTHEW SHABET
# function: make a pdf from a plot
# input: plots, name, w, h
# output: NA (generates the pdf)
make.pdf <- function(plots, name, w, h) {
  if (any(c("gg", "ggplot", "Heatmap") %in% class(plots))) {plots = list(plots)}
  pdf(file=name, width=w, height=h)
  lapply(plots, print)
  dev.off()
}

# function: add MapMyCells output to seurat obj
# input: obj (with a meta.data field called type)
# output: obj
mmc_metadata = function(obj, mmc_df) {
  obj$mmc_class = mmc_df$Class_label_label
  obj$mmc_class_prob = mmc_df$Class_label_bootstrapping_probability
  obj$mmc_subclass = mmc_df$Subclass_label_label
  obj$mmc_subclass_prob = mmc_df$Subclass_label_bootstrapping_probability
  obj$mmc_group = mmc_df$Group_label_label
  obj$mmc_group_prob = mmc_df$Group_label_bootstrapping_probability
  return(obj)
}

add_zone_labels = function(obj, mmc_df) {
  obj$zones_WM_GM_lt = mmc_df$zones_WM_GM_name
  obj$zones_WM_GM_prob = mmc_df$zones_WM_GM_bootstrapping_probability
  obj$zones_NAc_CP_lt = mmc_df$zones_NAc_CP_name
  obj$zones_NAc_CP_prob = mmc_df$zones_NAc_CP_bootstrapping_probability
  obj$zones_WM_C_lt = mmc_df$zones_WM_C_name
  obj$zones_WM_C_prob = mmc_df$zones_WM_C_bootstrapping_probability
  return(obj)
}

load_h5ad = function(obj_path, counts_assay = 'X') {
  print('Read as sce...')
  sce <- readH5AD(obj_path, reader = 'python', verbose = T)
  print('Create Seurat object...')
  seurat_obj <- as.Seurat(
    sce,
    counts = counts_assay,   # use the “counts” assay for raw counts
    data   = NULL        # NULL means don’t override the default “data” slot
  )
  seurat_obj %<>% RenameAssays(., assay.name = 'originalexp', new.assay.name = 'RNA')
  rm(sce)
  gc()
  return(seurat_obj)
}


