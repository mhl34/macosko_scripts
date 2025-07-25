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


# function: showing cells that are present in the sub_obj
# input: obj, sub_obj
# output: plot
selective_showing <- function(obj, label, labeltype = 'mmc_group', 
                              highlight_color = "firebrick", reduction = 'spatial') {
  # 1) build the subset identity
  obj$subset_identity <- ifelse(
    obj[[labeltype]] == label,
    label,
    "Other"
  )
  
  # 2) make it a factor with levels Other, then label
  obj$subset_identity <- factor(
    obj$subset_identity,
    levels = c("Other", label)
  )
  
  # 3) set Idents so DimPlot picks it up
  Idents(obj) <- obj$subset_identity
  
  # 4) color vector: Other = grey, label = highlight_color
  cols <- c(Other = "lightgrey", setNames(highlight_color, label))
  
  # 5) plot
  plot = DimPlot(
    obj,
    reduction = reduction,
    group.by  = "subset_identity",
    cols      = cols,
    order     = TRUE
  ) + 
    coord_fixed(ratio=1) +
    labs(x = "x (um)", y = "y (um)") + 
    theme_void() +
    NoLegend() + 
    ggtitle(paste0(label))
  # + theme(
  # plot.title = element_text(size = 0, face = "bold")  # **Bold and reduce title size**
  # )
  
  return(plot)
}


# function: plot separate graphs for each cluster cleanly (with max and min)
# input: obj, field (what to split on)
# output: plot (ggplot object)
plot_split = function(obj, field = 'seurat_clusters', reduction = 'spatial', highlight_colors = c('firebrick')) {
  plots = list()
  for (type in levels(obj@meta.data[[field]])) {
    if (len(highlight_colors) == 1) {
      highlight_color = highlight_colors[[1]]
    } else {
      highlight_color = highlight_colors[[type]]
    }
    plots[[type]] = selective_showing(obj, type, labeltype = field, reduction = reduction, highlight_color = highlight_color)
  }
  plot = wrap_plots(plots) + plot_layout(guides = "collect")  
  return(plot)
}

# function: plots proportion of cells (can title plot and also the column name to select for)
# input: obj, title (optional), type (optional)
# output: list(plot, frequency table)
plot_proportion = function(obj, title="", type = 'type') {
  freq_table <- table(obj[[type]])
  percentages <- round(100 * freq_table / sum(freq_table), 1)
  par(mar = c(10, 4, 4, 2) + 0.1)
  bp <- barplot(percentages,
                main = paste(title, "(Total Number of Cells:", sum(freq_table), ")"),
                ylab = "Percentage",
                col = "skyblue",
                border = "black",
                ylim = c(0, 100),
                las = 2,
                cex.names = 0.7)
  text(x = bp, y = percentages, labels = paste0(percentages, "%"), pos = 3, cex = 0.8, col = "black")
  return(list(plot = bp, table = freq_table))
}


# function: plotting a heatmap

make_proportional_heatmap <- function(df, row_col, col_col, digits = 2) {
  # Ensure the input columns exist
  if (!all(c(row_col, col_col) %in% colnames(df))) {
    stop("Both column names must exist in the dataframe.")
  }
  
  # Create a contingency table and compute row-wise proportions
  tab <- df %>%
    dplyr::count(.data[[row_col]], .data[[col_col]]) %>%
    group_by(.data[[row_col]]) %>%
    mutate(proportion = n / sum(n)) %>%
    ungroup()
  
  # Rename for plotting
  names(tab)[1:2] <- c("Row", "Col")  # Standardize for plotting
  
  # Add formatted label for text
  tab$label <- format(round(tab$proportion, digits), nsmall = digits)
  
  # Plot the heatmap with text
  ggplot(tab, aes(x = Col, y = Row, fill = proportion)) +
    geom_tile(color = "white") +
    # geom_text(aes(label = label), size = 4) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(x = col_col, y = row_col, fill = "Proportion") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

CleanFeaturePlot <- function(obj, gene_vector, reduction = 'spatial', label = F, ncols = 3, size = 12) {
  plots = list()
  for (i in seq_along(1:len(gene_vector))) {
    plots[[i]] <- FeaturePlot_scCustom(obj, gene_vector[i], reduction = reduction, label = label) +
      theme_void() +
      NoLegend() +
      coord_fixed() +
      theme(
        plot.tag = element_blank(),
        plot.title = element_text(size = 10, hjust = 0.5, lineheight = 1.1)  # smaller, centered
      )
  }
  plot <- wrap_plots(plots, guides = "collect") +
    plot_layout(ncol = ncols) +              # optional: force 5 per row
    plot_annotation(tag_levels = NULL)  # prevents patchwork from adding labels
  return(plot)
}

CleanDimPlot <- function(obj, sample_names, feature = 'donor_id', reduction = 'spatial', grouptype = 'seurat_clusters', label = F) {
  plots = list()
  for (i in seq_along(1:len(sample_names))) {
    sub_obj = obj[, obj[[feature]] == sample_names[i]]
    plots[[sample_names[i]]] <- DimPlot_scCustom(sub_obj, group.by = grouptype, reduction = reduction, label = label) +
      theme_void() +
      NoLegend() +
      coord_fixed() +
      theme(
        plot.tag = element_blank(),     # removes top-left tag label
        plot.margin = margin(1, 1, 1, 1)  # minimal margin around each plot
      )
  }
  plot <- wrap_plots(plots, guides = "collect") +
    plot_layout(ncol = 5) +              # optional: force 5 per row
    plot_annotation(tag_levels = NULL)  # prevents patchwork from adding labels
  return(plot)
}
