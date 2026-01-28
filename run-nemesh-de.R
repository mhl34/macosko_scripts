suppressWarnings(suppressMessages(library(magrittr)))
suppressWarnings(suppressMessages(library(getopt)))
suppressWarnings(suppressMessages(library(Seurat)))
suppressWarnings(suppressMessages(library(devtools)))
suppressWarnings(suppressMessages(library(Matrix)))
suppressWarnings(suppressMessages(library(ashr)))
suppressWarnings(suppressMessages(library(getopt)))
suppressWarnings(suppressMessages(library(ggplot2)))
suppressWarnings(suppressMessages(library(limma)))
suppressWarnings(suppressMessages(library(Matrix)))
suppressWarnings(suppressMessages(library(qs)))
suppressWarnings(suppressMessages(library(RhpcBLASctl)))
suppressWarnings(suppressMessages(library(TRADEtools)))
suppressWarnings(suppressMessages(library(tools)))
suppressWarnings(suppressMessages(source('~/helpers/object_helpers.R')))
suppressWarnings(suppressMessages(source('~/zonation/nemesh_differential_expression/differential_expression_new.R')))

export_seurat_to_DGEList <- function(
    seurat_obj,
    dir,
    prefix = "dge",
    group_col = "donor_id",
    zone_id = NULL,
    diff_expr_flag = TRUE  # <-- new argument
) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  
  if (!is.null(zone_id)) {
    message('add zone factor')
    seurat_obj$zone_label = paste0('zone_',unname(unlist(seurat_obj[[zone_id]])))
  }
  
  counts_file  <- file.path(dir, paste0(prefix, "_counts.tsv.gz"))
  samples_file <- file.path(dir, paste0(prefix, "_samples.tsv.gz"))
  
  # ---- Extract counts ----
  assay_name <- Seurat::DefaultAssay(seurat_obj)
  counts_mat <- Seurat::GetAssayData(seurat_obj, assay = assay_name, slot = "counts")
  counts_mat <- round(as.matrix(counts_mat))
  
  # ---- Extract metadata ----
  meta <- seurat_obj@meta.data
  
  # Align counts and metadata
  common_cells <- intersect(colnames(counts_mat), rownames(meta))
  if (length(common_cells) == 0)
    stop("No overlapping cell/sample IDs between counts and metadata.")
  counts_mat <- counts_mat[, common_cells, drop = FALSE]
  meta <- meta[common_cells, , drop = FALSE]
  
  # Add a SampleID and optional flag column
  meta$SampleID <- rownames(meta)
  meta$differential_expression <- diff_expr_flag  # <-- here
  
  # ---- Write counts ----
  counts_df <- data.frame(GeneID = rownames(counts_mat), counts_mat, check.names = FALSE)
  gz_counts_con <- gzfile(counts_file, "w")
  utils::write.table(counts_df, gz_counts_con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(gz_counts_con)
  
  # ---- Write samples ----
  gz_samples_con <- gzfile(samples_file, "w")
  utils::write.table(meta, gz_samples_con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(gz_samples_con)
  
  message("✅ Export complete:")
  message(" - Counts:  ", normalizePath(counts_file))
  message(" - Samples: ", normalizePath(samples_file))
  
  invisible(list(counts = counts_file, samples = samples_file))
}

spec <- matrix(c(
  'dir-label', 'd', 1, 'character',
  'label', 'l', 1, "character",
  'num-threads', 'n', 1, 'integer'
), byrow = TRUE, ncol = 4)

opt <- getopt(spec)

if (! is.null(opt[['num-threads']])){
  blas_set_num_threads(opt[['num-threads']])
}

dir_label = opt[['dir-label']]
label = opt[['label']]
num_threads = opt[['num-threads']]

in_dir = paste0('~/zonation/TWI_analysis/', dir_label, '/', label, '/', label, '.h5ad')
out_dir = paste0('~/zonation/TWI_analysis/', dir_label, '/', label, '/', label, '.qs')
obj1 = load_h5ad(in_dir)
obj1$cell_type = label
qsave(obj1, out_dir)
obj1 = qread(out_dir)

obj1$pct_intronic = obj1$pct_intronic * 100
obj1$pct_mt = obj1$pct_mt * 100

export_seurat_to_DGEList(obj1, paste0('~/zonation/TWI_analysis/', dir_label, '/', label), prefix = label, group_col =  'donor_id')

data_dir=paste0('~/zonation/TWI_analysis/', dir_label, '/', label)
data_name=label
randVars=c("donor_id","village")
interaction_var = 'DV_label_permuted'
dir.create(paste0('~/zonation/TWI_analysis/', dir_label, '_contrasts/', label))
contrast_file=paste0('~/zonation/TWI_analysis/', dir_label, '_contrasts/', label, "/differential_expression_",dir_label,"_contrasts_age.txt")
dir.create(dirname(contrast_file), recursive = TRUE, showWarnings = FALSE)
con <- file(contrast_file, "w")
writeLines("contrast_name\tvariable\treference_level\tcomparison_level\tbaseline_region\nage\tage\tNA\tNA\tDorsolateral\n", con)
close(con)
out_tag = 'out'
outPDF_path = paste0('~/zonation/TWI_analysis/', dir_label, '/', label, '/', out_tag, '/', label,'_output.pdf')
result_path = paste0('~/zonation/TWI_analysis/', dir_label, '/', label, '/', out_tag, '/')
fixedVars=c("DV_label_permuted", "age", "PC1", "PC2", "PC3", "PC4", "PC5", "pct_intronic", "pct_mt", "frac_contamination", "logumi", "imputed_sex", "single_cell_assay", "biobank")
dir.create(paste0(data_dir, '/', out_tag, '/'))

bican_de_differential_expression(
  data_dir=data_dir, 
  data_name=data_name, 
  contrast_file=contrast_file,
  randVars=randVars, 
  fixedVars=fixedVars,
  interaction_var=interaction_var,
  absolute_effects = TRUE,
  outPDF=outPDF_path, 
  result_dir=result_path,
  n_cores = num_threads)
