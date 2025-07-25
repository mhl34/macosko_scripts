from cnmf import cNMF, Preprocess
import anndata as ad
import numpy as np
import scanpy as sc
import os
import multiprocessing
import gzip
import scipy.io
import scipy.sparse as sp
import argparse

def run_factorize(worker_i):
    cnmf_obj.factorize(worker_i=worker_i, total_workers=ncores)

def get_args():
    parser = argparse.ArgumentParser(description='run cNMF on adata')
    parser.add_argument("-i", "--in_dir", help="input data folder", type=str, default=".")
    parser.add_argument("-o", "--out_dir", help="output data folder", type=str, default=".")
    parser.add_argument("-r", "--regex", help="regex match for file", type=str)
    parser.add_argument("-n", "--ncores", help="number of cores for multiprocessing", type=int, default=8)
    parser.add_argument("-cl", "--cnmf_lower", help="set cNMF lower bound", type=int, default = 3)
    parser.add_argument("-cu", "--cnmf_upper", help="set cNMF upper bound", type=int, default = 20)
    parser.add_argument("-e", "--epochs", help="set number of iterations for cNMF", type=int, default = 100)
    parser.add_argument("-k", "--k_param", help="set the k (number of NMF factors)", type = int, default = 6)
    args, unknown = parser.parse_known_args()
    [print(f"WARNING: unknown command-line argument {u}") for u in unknown]
    return args

args = get_args()

in_dir = args.in_dir
out_dir = args.out_dir
ncores = args.ncores
cnmf_lower = args.cnmf_lower
cnmf_upper = args.cnmf_upper
epochs = args.epochs
K = args.k_param
regex_pattern = args.regex

def find_local_maxima(arr, skip_plateau=True, first_only=False, prefer_rightmost_plateau=True):
    n = len(arr)
    if n < 3:
        return None if first_only else []

    start = 1
    if skip_plateau:
        while start < n and arr[start] == arr[start - 1]:
            start += 1
        start = min(start, n - 2)

    maxima = []
    i = start
    while i < n - 1:
        # Plateau start
        if arr[i] == arr[i + 1]:
            plateau_start = i
            while i < n - 1 and arr[i] == arr[i + 1]:
                i += 1
            plateau_end = i

            # Check if it's higher than surroundings
            left = arr[plateau_start - 1] if plateau_start - 1 >= 0 else float('-inf')
            right = arr[plateau_end + 1] if plateau_end + 1 < n else float('-inf')
            if arr[plateau_start] > left and arr[plateau_end] > right:
                max_idx = plateau_end if prefer_rightmost_plateau else plateau_start
                if first_only:
                    return max_idx
                maxima.append(max_idx)
        elif arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if first_only:
                return i
            maxima.append(i)
        i += 1

    return maxima if not first_only else None


# in_dir = '/broad/macosko/leematth/projects/Jinoh/illumina/e12'
# out_dir = '/broad/macosko/leematth/projects/Jinoh/illumina/e12/cNMF_results'

# /broad/macosko/leematth/projects/Jinoh/sofov1_tf_subset.h5ad

# adata = ad.read_h5ad(os.path.join(in_d-ir, 'e12_merge_section_0_tfs_sparkx.h5ad'))
for filename in os.listdir(in_dir): 
    if regex_pattern not in filename or '.h5ad' not in filename:
        continue
    filename_base = filename[:-5]
    print(filename_base)
    adata = ad.read_h5ad(os.path.join(in_dir, filename))
    print('load...')
    cnmf_obj = cNMF(output_dir=out_dir, name=filename_base)
    print('prepare...')
    print('preparing matrix')
    # 1. Write matrix.mtx.gz
    if "matrix.mtx.gz" not in os.listdir(os.path.join(out_dir, filename_base)):
        mtx = adata.X.T.copy()
        # mtx = adata.layers['raw']
        mtx_path = os.path.join(out_dir,filename_base, "matrix.mtx")
        scipy.io.mmwrite(mtx_path, mtx)
        with open(mtx_path, "rb") as f_in, gzip.open(mtx_path + ".gz", "wb") as f_out:
            f_out.writelines(f_in)
        os.remove(mtx_path)
    
    # 2. Write barcodes.tsv.gz
    if "barcodes.tsv.gz" not in os.listdir(os.path.join(out_dir, filename_base)):
        bar_path = os.path.join(out_dir,filename_base, "barcodes.tsv.gz")
        with gzip.open(bar_path, "wt") as f:
            for bc in adata.obs_names:
                f.write(bc + "\n")
    
    # 3. Write features.tsv.gz (gene_id and gene_name)
    if "features.tsv.gz" not in os.listdir(os.path.join(out_dir, filename_base)):
        feat_path = os.path.join(out_dir,filename_base, "features.tsv.gz")
        # adata.var["feature_types"] = adata.var["gene_symbols"]
        with gzip.open(feat_path, "wt") as f:
            for gid, sym in zip(adata.var_names, adata.var_names):
                f.write(f"{gid}\t{sym}\tGene Expression\n")
    
    print("Generated 10x files in:", out_dir)
    cnmf_obj.prepare(counts_fn=os.path.join(out_dir,filename_base, "matrix.mtx.gz"), components=np.arange(cnmf_lower, cnmf_upper), n_iter=epochs, seed=14, num_highvar_genes=adata.shape[1])
    print('pool...')
    with multiprocessing.Pool(processes=ncores) as pool:
        pool.map(run_factorize, range(ncores))
    print('combine...')
    cnmf_obj.combine()
    print('k selection plot...')
    cnmf_obj.k_selection_plot()
    selection_stats = np.load(os.path.join(out_dir,filename_base,f'{filename_base}.k_selection_stats.df.npz'))
    
    maxima = find_local_maxima(selection_stats['data'][0:,2].tolist(), first_only = False)
    if maxima != None:
        for maximum in maxima:
            idx = maximum + cnmf_lower if len(maxima) != 0 else cnmf_lower
            if idx > cnmf_upper - 4:
                continue
            print(f'selected K: {idx}')
            print('consensus...')
            cnmf_obj.consensus(k=idx, density_threshold=0.05)
            print('load results...')
            usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=idx, density_threshold=0.05)
            
            usage.to_csv(os.path.join(out_dir,filename_base, f'usage_K_{idx}.csv'))
            spectra_scores.to_csv(os.path.join(out_dir,filename_base, f'spectra_scores_K_{idx}.csv'))
            spectra_tpm.to_csv(os.path.join(out_dir,filename_base, f'spectra_tpm_K_{idx}.csv'))
            top_genes.to_csv(os.path.join(out_dir,filename_base, f'top_genes_K_{idx}.csv'))
    else:
        idx = cnmf_lower
        print(f'selected K: {idx}')
        print('consensus...')
        cnmf_obj.consensus(k=idx, density_threshold=0.05)
        print('load results...')
        usage, spectra_scores, spectra_tpm, top_genes = cnmf_obj.load_results(K=idx, density_threshold=0.05)
        
        usage.to_csv(os.path.join(out_dir,filename_base, f'usage_K_{idx}.csv'))
        spectra_scores.to_csv(os.path.join(out_dir,filename_base, f'spectra_scores_K_{idx}.csv'))
        spectra_tpm.to_csv(os.path.join(out_dir,filename_base, f'spectra_tpm_K_{idx}.csv'))
        top_genes.to_csv(os.path.join(out_dir,filename_base, f'top_genes_K_{idx}.csv'))
