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
suppressWarnings(suppressMessages(source('/broad/macosko/leematth/helpers/object_helpers.R')))
suppressWarnings(suppressMessages(source('/broad/macosko/leematth/projects/zonation/nemesh_differential_expression/differential_expression_new.R')))

spec <- matrix(c(
  'label', 'l', 1, "character",
  'num-threads', 'n', 1, 'integer'
), byrow = TRUE, ncol = 4)

opt <- getopt(spec)

if (! is.null(opt[['num-threads']])){
  blas_set_num_threads(opt[['num-threads']])
}

label = opt[['label']]
num_threads = opt[['num-threads']]

in_dir = paste0('/broad/macosko/leematth/projects/zonation/jackknifing_TRADE_d1_mat/pseudobulks/', label, '/', label, '.h5ad')
out_dir = paste0('/broad/macosko/leematth/projects/zonation/jackknifing_TRADE_d1_mat/pseudobulks/', label, '/', label, '.qs')
obj1 = load_h5ad(in_dir)
obj1$cell_type = label
qsave(obj1, out_dir)
obj1 = qread(out_dir)

obj1$pct_intronic = obj1$pct_intronic * 100
obj1$pct_mt = obj1$pct_mt * 100

export_seurat_to_DGEList(obj1, paste0('/broad/macosko/leematth/projects/zonation/jackknifing_TRADE_d1_mat/pseudobulks/', label), prefix = label, group_col =  'donor_id')

data_dir=paste0('/broad/macosko/leematth/projects/zonation/jackknifing_TRADE_d1_mat/pseudobulks/', label)
data_name=label
randVars=c("donor_id","village")
interaction_var = 'DV_label'
dir.create(paste0('/broad/macosko/leematth/projects/zonation/jackknifing_TRADE_d1_mat/contrasts/', label))
contrast_file=paste0('/broad/macosko/leematth/projects/zonation/jackknifing_TRADE_d1_mat/contrasts/', label, "/differential_expression_contrasts_age.txt")
dir.create(dirname(contrast_file), recursive = TRUE, showWarnings = FALSE)
con <- file(contrast_file, "w")
writeLines("contrast_name\tvariable\treference_level\tcomparison_level\tbaseline_region\nage\tage\tNA\tNA\tDorsolateral\n", con)
close(con)
out_tag = 'out'
outPDF_path = paste0('/broad/macosko/leematth/projects/zonation/jackknifing_TRADE_d1_mat/pseudobulks/', label, '/', out_tag, '/', label,'_output.pdf')
result_path = paste0('/broad/macosko/leematth/projects/zonation/jackknifing_TRADE_d1_mat/pseudobulks/', label, '/', out_tag, '/')
fixedVars=c("age", "PC1", "PC2", "PC3", "PC4", "PC5", "pct_intronic", "pct_mt", "frac_contamination", "logumi", "imputed_sex", "single_cell_assay", "biobank")
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
