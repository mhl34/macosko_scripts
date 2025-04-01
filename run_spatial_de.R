library(NMF)
library(qs)
library(Seurat)
library(harmony)
library(future)
library(future.apply)
library(parallel)
library(parallelly)
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
library(ggpubr)

# Enable progress bar
# handlers(global = TRUE)  # Use default progress bar handler
options(future.globals.maxSize = 300 * 1024 * 1024 * 1024)
availCores = parallelly::availableCores()
plan(multisession, workers=availCores)
# setDTthreads(parallelly::availableCores())

stopifnot(file.exists("spatial_de/helpers.R"))
source('spatial_de/helpers.R')

args = commandArgs(trailingOnly=TRUE)
if (length(args) == 1) {
  obj_path <- args[[1]]
  out_path <- "./"
} else if (length(args) == 2) {
  obj_path <- args[[1]]
  out_path <- args[[2]]
} else {
  stop("Usage: Rscript run_spatial_de.R obj_path (out_path)", call. = FALSE)
}

# TEST
obj = qread(obj_path)

print(sprintf("obj path: %s", obj_path))
print(sprintf("out path: %s", out_path))

print('Remove mito...')
obj %<>% remove_mito
obj %<>% cleaning

# print("Run harmony...")
# run_harmony(obj, key = 'library')

# print('Process object...')
# obj %<>% process

spatial_label = ifelse('spatial' %in% Reductions(obj), 'spatial', 'Spatial')

# get embeddings
print('Get placed cells...')
if (!(spatial_label %in% Reductions(obj))) {
  stop(sprintf(
    "The Seurat object does not contain spatial embeddings as a dimensional reduction called 'spatial'"
  ))
}
spatial_df = as.data.frame(Embeddings(obj, reduction = spatial_label))
s1_label = colnames(spatial_df)[1]
s2_label = colnames(spatial_df)[2]
spatial_mask = !is.na(spatial_df[[s1_label]])
if (sum(!spatial_mask) > 0) {
  warning(sprintf("Removing %d cells without spatial coordinates", sum(!spatial_mask)))
}
spatial_df = spatial_df[spatial_mask,]
obj = obj[,spatial_mask]

# get expression matrix
print('Running SPARK-X...')
obj_data = as.matrix(obj@assays$RNA$counts)
obj_data_cell_norm = sweep(obj_data, 2, colSums(obj_data), "/")
scaled_spatial_df = as.data.frame(scale(spatial_df))
sparkx_res <- sparkx(obj_data, scaled_spatial_df, numCores=availCores, option="mixture")
sparkx_res_sort <- sparkx_res$res_mtest[order(sparkx_res$res_mtest$adjustedPval),]

print('Generate Mask...')
mask = rownames(sparkx_res_sort[sparkx_res_sort$adjustedPval < 0.0005,])
print(sprintf("Indentified %d spatially significant genes out of %d", length(mask), dim(sparkx_res_sort)[1]))
if (length(mask) < 9) {
  stop(sprintf(
    "Too few spatially varying genes found (< 9)"
  ))
}
print('Filter for Spatial Significant Genes...')
cell_mask = rownames(obj_data[which(rowSums(obj_data > 0) > dim(obj_data)[2] * 0.01),])
obj_data_masked = obj_data[intersect(mask,cell_mask),]
obj_data_cell_norm = sweep(obj_data_masked, 2, colSums(obj_data_masked), "/")
loess_gene_label = rownames(obj_data[intersect(mask,cell_mask),])
sparkx_res_filtered = sparkx_res_sort[intersect(mask,cell_mask),]

print('Run LOESS...')
loess_res = future_apply(obj_data_cell_norm, MARGIN = 1, FUN = function(row) {
  df <- data.frame("x" = as.vector(spatial_df[[s1_label]]), "y" = as.vector(spatial_df[[s2_label]]), "z" = row, check.names = TRUE)
  loess <- loess(z ~ x + y, data = df, span = 0.1)
  # Return the predicted vector for each row
  predict(loess)
})
loess_res = t(loess_res)
loess_res[loess_res < 0] = 0

print('Run NMF (9 Features)...')
rank = 9
nmf_res = RcppML::nmf(loess_res, rank, tol = 1e-10, maxit = 20000, n_threads = availCores)

w = as.data.frame(nmf_res@w)
h = nmf_res@h
nmf_cols = lapply(1:dim(h)[1], FUN=function(i){sprintf("Factor %d", i)})
colnames(w) = nmf_cols

print('Produce Plots...')
plots = lapply(1:rank, function(idx) {
  expression = nmf_res@h[idx,]
  nmf_col = nmf_cols[idx]
  ggplot() + geom_point(aes(x = spatial_df[[s1_label]], y = spatial_df[[s2_label]],color = expression)) +
    scale_color_viridis(option = "inferno") +
    theme_minimal() + 
    coord_fixed(ratio=1) +
    labs(x = "x (um)", y = "y (um)") + 
    theme(
      axis.text = element_text(size = 6),
      axis.title = element_text(size = 10),
      # axis.text.x = element_text(angle = 90, hjust = 1)
    ) + 
    theme(
      plot.title = element_text(face = "bold")  # **Bold and reduce title size**
    ) + 
    plot_annotation(title = nmf_col)
})

w$entropy_vals <- apply(w, 1, function(row) {
  prob_dist <- row / sum(row)
  entropy::entropy(prob_dist) 
})
w$genes = rownames(w)
w = w[order(w$entropy_vals),]
w$argmax <- colnames(w)[apply(w[, 1:rank], 1, which.max)]
w_group <- w %>% group_by(argmax) %>% slice_head(n = 16) %>% ungroup() %>% as.data.frame

stopifnot(file.exists(obj_path))
if (!dir.exists(out_path)) {
  dir.create(out_path)
}

k2 = 16
for (idx in 1:rank) {
  print(idx)
  nmf_col = nmf_cols[[idx]]
  plot1 = plots[idx][[1]]
  w_group_sub = w_group[w_group$argmax == nmf_col,]
  w_group_sub = w_group_sub[order(w_group_sub$entropy_vals, -w_group_sub[[nmf_col]]),]
  w_group_sub$x_pos = dim(w_group_sub)[1]:1 / dim(w_group_sub)[1]
  gene_list = w_group_sub$genes
  plot2 = ggscatter(w_group_sub, x = "x_pos", y = nmf_col) +
    annotate("text",
             x = 1.08,
             y = seq(from = max(w_group_sub[[nmf_col]]), to = 0, length.out = length(w_group_sub$genes)),
             label = w_group_sub$genes,
             size = 2,
             hjust = 0) +
    theme_bw() +
    # scale_x_continuous(limits = c(NA, 1.0)) +
    ggplot2::theme_bw() +
    ggplot2::theme(
      axis.ticks.x = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_blank(),
      axis.title.x = ggplot2::element_blank(),
      axis.title.y = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor.x = ggplot2::element_blank(),
      legend.position = "none"
    ) +
    ggplot2::coord_cartesian(
      xlim = c(0, 1),
      clip = "off"
    ) +
    ggplot2::theme(plot.margin = grid::unit(c(1, 4, 1, 1), "lines")) +
    ggtitle("Gene Loadings")
  plot_top = plot_grid(plot1, plot2, ncol=2, rel_widths = c(0.6,0.4))
  plotlist_3 = lapply(gene_list, function(gene) {
    FeaturePlot_scCustom(obj, gene, reduction = spatial_label, coord.fixed = 1) +
      labs(x = "x (um)", y = "y (um)") + 
      theme_void() +
      NoLegend() + 
      theme(
        plot.title = element_text(size = 10, face = "bold")  # **Bold and reduce title size**
      )
  })
  plot_bottom = wrap_plots(plotlist_3) + plot_layout(guides = "collect")  
  plot = plot_grid(plot_top, plot_bottom, ncol=1, rel_heights=c(0.4,0.6))
  make.pdf(plot, sprintf(paste0(out_path, 'nmf_%d.pdf'), idx), 7, 8)
}

print('Produce output...')
pdfs = sprintf(paste0(out_path, 'nmf_%d.pdf'), 1:rank)
qpdf::pdf_combine(input=pdfs, output=paste0(out_path,"spatial_de_result.pdf"))
file.remove(pdfs)

print('Run KNN...')
k = 45
matrix_knn <- find_knn(loess_res / rowSums(loess_res), k = k, metric = 'hellinger', n_threads = availCores)
gene_neighbors <- as.data.frame(matrix(row_names[matrix_knn$idx], nrow = nrow(matrix_knn$idx), byrow = TRUE))
write.csv(gene_neighbors, paste0(out_path, "spatial_genes.csv"))
