library(Seurat)
library(Matrix)
library(Seurat)
install.packages(SeuratData)
library(SeuratData)
library(SeuratWrappers)
library(ggplot2)
library(dplyr)
library(patchwork)
library(magrittr)
library(graphics)
library(qs)

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly = T)
ref_name <- args[1]
query_name <- args[2]

convert.folder.to.seurat = function(folder.path, dest.path) {
  if (!dir.exists(folder.path)) {
    stop(paste0("Error! The folder: ", folder.path, " does not exist."))
  }
  
  counts.path = paste0(folder.path, "/counts.mtx")
  cell.meta.path = paste0(folder.path, "/cell_meta.csv")
  feature.meta.path = paste0(folder.path, "/feature_meta.csv")
  
  print(paste0("Counts Mat: ", counts.path))
  print(paste0("Cell Meta: ", cell.meta.path))
  print(paste0("Feature Meta: ", feature.meta.path))
  
  if (!file.exists(counts.path)) {
    stop("Counts DNE.")
  }
  if (!file.exists(cell.meta.path)) {
    stop("cell.meta.path DNE.")
  }
  if (!file.exists(feature.meta.path)) {
    stop("feature.meta.path DNE.")
  }
    
  counts.mat = Matrix::readMM(counts.path)
  counts.mat = Matrix::t(counts.mat)
  cell.meta = read.csv(cell.meta.path)
  row.names(cell.meta) = cell.meta$X
  feature.meta = read.csv(feature.meta.path)
  gene.names = feature.meta$gene.name
  
  if (NCOL(counts.mat) != NROW(cell.meta)) {
    print(paste0("NCOL Count: ", NCOL(counts.mat)))
    print(paste0("NROW cell.meta: ", NROW(cell.meta)))
    stop("NCOL counts must equal NROW cell.meta")
  }
  
  if (NROW(counts.mat) != length(gene.names)) {
    print(paste0("NROW Count: ", NROW(counts.mat)))
    print(paste0("length gene.names: ", length(gene.names)))
    stop("n genes must equal rows of matrix")
  }
  
  colnames(counts.mat) = cell.meta$X
  row.names(counts.mat) = gene.names
  
  seurat.obj = CreateSeuratObject(counts.mat, assay = "RNA")
  print(seurat.obj)
  seurat.obj@meta.data = cbind(seurat.obj@meta.data, cell.meta)
  
  print(paste0("Saving obj to ", dest.path))
  qsave(seurat.obj, dest.path)
  
}

process_data = function(obj) {
  obj <- obj %>%
    Seurat::NormalizeData() %>%
    Seurat::FindVariableFeatures() %>%
    Seurat::ScaleData() %>%
    Seurat::RunPCA(npcs=50, verbose=F) %>%
    Seurat::FindNeighbors(dims=1:30) %>%
    Seurat::FindClusters(resolution = 2) %>%
    Seurat::RunUMAP(dims=1:30, verbose=F)
}

convert.folder.to.seurat(ref_name, paste0(ref_name, "/ref.qs"))
convert.folder.to.seurat(ref_name, paste0(ref_name, "/query.qs"))

print("read in data")
ref_obj <- qread(paste0(ref_name, "/ref.qs"))
query_obj <- qread(paste0(ref_name, "/query.qs"))

print("process data")
ref_obj %<>% process_data()
query_obj %<>% process_data()

ref_obj$label <- ref_name
query_obj$label <- query_name

print("integrate layers")
adata.combined <- IntegrateLayers(
  object = adata.combined, method = CCAIntegration,
  orig.reduction = "pca", new.reduction = "integrated.cca"
)

print("process combined data")
adata.combined <- adata.combined %>%
  Seurat::NormalizeData() %>%
  Seurat::FindVariableFeatures() %>%
  Seurat::ScaleData() %>%
  Seurat::RunPCA(npcs=50, verbose=F) %>%
  Seurat::FindNeighbors(reduction = "integrated.harmony", dims=1:30) %>%
  Seurat::FindClusters(resolution = 2) %>%
  Seurat::RunUMAP(dims=1:30, reduction = "integrated.harmony", verbose=F)

print("save combined object")
qsave(adata.combined, paste0(query_name, "/combined.qs"))

print("make predictions")
adata.ref <- subset(adata.combined,  subset = label == ref_name)
adata.query <- subset(adata.combined,  subset = label == query_name)

adata.query <- NormalizeData(adata.query)
adata.query.anchors <- FindTransferAnchors(reference = adata.ref, query = adata.query, dims = 1:30, reference.reduction = "pca")

clusterNmPredictions <- TransferData(anchorset = adata.query.anchors, refdata = adata.ref$ClusterNm, dims = 1:30)
cladeNamePredictions <- TransferData(anchorset = adata.query.anchors, refdata = adata.ref$clade_name, dims = 1:30)

adata.query <-  AddMetaData(adata.query, metadata = clusterNmPredictions)

write.csv(adata.query@meta.data, paste0(query_name, "/combined_pred.csv"))
