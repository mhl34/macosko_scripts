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
library(scCustomize)

# markergenes <- list(
#   astrocytes = c("GFAP", "ALDH1L1", "AQP4", "S100B", "GLUL", "SOX9", "GJA1", "SLC1A3"),
#   endothelial_cells = c("PECAM1", "VWF", "CDH5", "CLDN5", "ESAM", "TIE1", "MCAM", "ENG"),
#   microglia = c("CX3CR1", "P2RY12", "TMEM119", "ITGAM", "CD68", "CSF1R", "AIF1", "SALL1"),
#   neurons = c("RBFOX3", "MAP2", "TUBB3", "SYN1", "NEFL", "NRXN1", "GRIN1", "SLC17A7"),
#   oligodendrocytes = c("MBP", "MOG", "PLP1", "MAG", "OLIG1", "OLIG2", "CLDN11", "GPR17"),
#   OPCs = c("PDGFRA", "CSPG4", "NKX2-2", "SOX10", "OLIG2", "VCAN", "LHFPL3", "BCAN")
# )

markergenes=list(
  OL=c("OLIG1", "MOG", "MBP", "PLP1"), #"TRF","PALM2","PPM1G","LGALS1" # OlOp <- c('MBP', 'PLP1', 'MOBP', 'OLIG1', 'OLIG2')
  OPC=c("CSPG5", "VCAN", "OLIG2", "CSPG4"),
  AS=c("AQP4", "GFAP", "GINS3", "ALDH1A1"),
  EN=c("BSG", "DCN", "RGS5", "FLT1"),
  MGL=c("C1QA","C1QB","CSF1R","CX3CR1","ITGAX","MGAM","PLAT","TMEM119","TREM2", "CD74"),
  InN=c('GAD1', 'GAD2', 'SST', 'PVALB', 'LHX6', 'LAMP5', 'VIP', 'HTR3A', "RELN", "NOG"),
  ExN=c('SLC17A7', 'SLC17A6', 'SLC17A8'),
  # DA=c("SLC6A3", "TH", "NR4A2"),
  VASC=c('CLDN5', 'FLT1', 'DCN')
)

# function: return a Seurat Object with cell type annotations in "type"
# input: obj (with a meta.data field called type)
# output: obj
anno.type <- function(obj, mouse=F) {
  if(!('type' %in% colnames(obj@meta.data))) {
    obj$type = 'unknown'
  }
  if(mouse){markergenes=mousemarkergenes}
  while(TRUE) {
    for (i in 1:length(markergenes)) {
      if (!"unknown" %in% unique(obj$type)) {print("returning obj") ; return(obj)}
      p = FeaturePlot_scCustom(subset(x=obj,subset=type=="unknown"), reduction="umap", features=markergenes[[i]], label=T) & NoLegend() & theme(aspect.ratio = 1) & theme(axis.line=element_blank(),axis.text.x=element_blank(),axis.text.y=element_blank(),axis.ticks=element_blank(),axis.title.x=element_blank(),axis.title.y=element_blank(),legend.position="none",panel.background=element_blank(),panel.border=element_blank(),panel.grid.major=element_blank(),panel.grid.minor=element_blank(),plot.background=element_blank())
      print(p)
      ct = names(markergenes)[[i]] ; print(g("Input {ct} clusters"))
      res = readline()
      if(nchar(res)<1) {next}
      if(res=="all"){obj$type[obj$type=="unknown"]=ct ; print("returning obj") ; return(obj)}
      if(sum(unlist(lapply(letters,function(x){grepl(x,res)})) | unlist(lapply(LETTERS,function(x){grepl(x,res)})))>0){print("returning obj") ; return(obj)}
      res = res %>% str_split_1(" ") %>% as.integer
      obj$type[obj$seurat_clusters %in% res] = ct
    }
  }
}
