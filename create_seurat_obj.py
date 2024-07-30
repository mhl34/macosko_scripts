import argparse
import anndata 
import os

# functions
def anndata_to_folder(anndata_path, folder_out_path):
    """
    creates a folder with the temporary files necessary to make seurat obj
    """
    counts_path = os.path.join(folder_out_path, "counts.mtx")
    cell_meta_path = os.path.join(folder_out_path, "cell_meta.csv")
    gene_meta_path = os.path.join(folder_out_path, "feature_meta.csv")

    scipy.io.mmwrite(counts_path, adata.X)
    adata.obs.to_csv(cell_meta_path)
    adata.var.to_csv(gene_meta_path)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ref', dest = ref, help = 'reference anndata object')
parser.add_argument('-q', '--query', dest = query, help = 'query anndata object')
parser.add_argument('-rn', '--ref_name', dest = ref_name, help = 'reference sample name')
parser.add_argument('-qn', '--query_name', dest = query_name, help = 'query sample name')
args = parser.parse_args()

# ref stores the path to reference anndata object
# query stores the path to query anndata object
ref = anndata.read_h5ad(args.ref)
query = anndata.read_h5ad(args.query)
ref_name = args.ref_name
query_name = args.query_name

# create a save directory
ref_dir = f"{os.path.dirname(ref)}/{ref_name}"
if not os.path.exists(ref_dir):
    os.mkdir(ref_dir)
query_dir = f"{os.path.dirname(query)}/{query_name}"
if not os.path.exists(query_dir):
    os.mkdir(query_dir)

anndata_to_folder(ref, ref_dir)
anndata_to_folder(query, query_dir)