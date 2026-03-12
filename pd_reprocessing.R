suppressWarnings(suppressMessages(library(scCustomize)))
suppressWarnings(suppressMessages(library(dplyr)))
suppressWarnings(suppressMessages(library(qs)))
suppressWarnings(suppressMessages(library(magrittr)))
suppressWarnings(suppressMessages(library(Seurat)))
suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(tidyverse)))
suppressWarnings(suppressMessages(library(glue))) ; g=glue ; len=length

add_spatial = function(obj) {
  emb = obj@meta.data[,c("x","y")] ; colnames(emb) = c("d_1","d_2")
  obj[["spatial"]] <- CreateDimReducObject(embeddings = as.matrix(emb), key = "d_")
  return(obj)
}

process <- function(obj, res=0.8, n.epochs=NULL, vars.to.regress = NULL) {
  obj <- obj %>%
    Seurat::NormalizeData() %>%
    Seurat::FindVariableFeatures() %>%
    Seurat::ScaleData(vars.to.regress = vars.to.regress) %>%
    Seurat::RunPCA(verbose=F) %>%
    Seurat::FindNeighbors(dims=1:30, n.trees = 100, reduction = 'pca') %>%
    Seurat::FindClusters(n.start = 20, n.iter = 20, algorithm = 4, resolution = res) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F, n.epochs=n.epochs, reduction = 'pca')
}

# look for an cellbender dir (h5), dropsift (csv), seurat_dir (qs)
arguments <- OptionParser(
  usage = "Usage: Rscript .R cellbender_path dropsift_path seurat_path out_path",
  option_list = list()
) %>% parse_args(positional_arguments=4)

cellbender_path = arguments$args[[1]]  ; print(g("cellbender_path: {cellbender_path}"))
dropsift_path   = arguments$args[[2]]  ; print(g("dropsift_path:   {dropsift_path}"))
seurat_path     = arguments$args[[3]]  ; print(g("seurat_path:     {seurat_path}"))
out_path        = arguments$args[[4]]  ; print(g("out_path:        {out_path}"))

h5          = Read_CellBender_h5_Mat('/broad/macosko/leematth/projects/PD/mehrdad_reprocessing/sample/gene-expression/SI-TT-F1_out.h5')
dropsift    = read.csv('/broad/macosko/leematth/projects/PD/mehrdad_reprocessing/sample/dropsift_outputs/dropsift_output.csv')
seurat_old  = qread('/broad/macosko/leematth/projects/PD/mehrdad_reprocessing/sample/spatial-data/seurat.qs')
seurat_new  = CreateSeuratObject(counts = h5[,colnames(seurat_old@assays$RNA$counts)])

seurat_new@meta.data['logumi'] = log1p(seurat_new@meta.data['nCount_RNA'])

# clusters to keep pct_intronic, pct_mt, RNA_snn_res.0.8, seurat_clusters, x, y, dbscan_clusters, dbscan_score
meta_to_merge <- seurat_old@meta.data[,c("pct_intronic", "pct_mt", 
                                         "RNA_snn_res.0.8", "seurat_clusters", 
                                         "x", "y", "dbscan_clusters", "dbscan_score")]
seurat_new    <- AddMetaData(seurat_new, metadata = meta_to_merge)

rownames(dropsift)    <- dropsift$cell_barcode
dropsift$cell_barcode <- NULL
dropsift = dropsift[rownames(seurat_new@meta.data),]
# include a check
stopifnot(dim(dropsift)[1]     == dim(seurat_new)[2])
metadata_dropsift              = dropsift[,c('frac_contamination', 
                                             'empty_gene_module_score', 
                                             'is_cell', 'is_cell_prob')]
metadata_dropsift              = metadata_dropsift %>% 
  rename(is_cell_dropsift      = is_cell, 
         is_cell_dropsift_prob = is_cell_prob)
seurat_new_dropsift            = AddMetaData(seurat_new, metadata = metadata_dropsift)

seurat_new_dropsift %<>% add_spatial
seurat_new_dropsift %<>% process

qsave(seurat_new_dropsift, out_dir)

