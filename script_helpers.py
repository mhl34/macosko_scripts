import os
import re
from collections import Counter
import csv
import glob
import pandas as pd
from os.path import basename
from itertools import product
import sys
import os
import subprocess
root = "/broad/macosko/pipelines"

indexes_root = "https://raw.githubusercontent.com/MacoskoLab/Macosko-Pipelines/main/bcl-convert/indexes/"
NN = pd.read_csv(indexes_root+"SI-NN.csv")
NT = pd.read_csv(indexes_root+"SI-NT.csv")
TS = pd.read_csv(indexes_root+"SI-TS.csv")
TT = pd.read_csv(indexes_root+"SI-TT.csv")
ND7 = pd.read_csv(indexes_root+"ND7.csv")
ND5 = pd.read_csv(indexes_root+"ND5.csv")

# Matthew Shabet Helper Methods from Notebook
def podman_sbatch_wrapper(command, logpath, jobname, mem, cpus=8, time="24:00:00"):
    assert "'" not in command and '"' not in command # can they be escaped?
    
    podman_command = f"podman run --rm --init --pull never -v {root}:{root}:rw pipeline-image '{command}'"
    sbatch_params = f"-C container -o {logpath} -J {jobname} \
                      --mem {mem} -c {cpus} -t {time} \
                      --mail-user macosko-pipelines@broadinstitute.org --mail-type END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT"
    if int(mem[:-1]) > 500:
        sbatch_params = "--partition=hpcx_macosko " + sbatch_params
    else:
        sbatch_params = "--nodelist=slurm-bits-d[002-005] " + sbatch_params
    cmd = f'sbatch {sbatch_params} --wrap "{podman_command}"'
    
    return(cmd)

def fastq_check(directory):
    min_size = 24 * 1024  # 24 KB in bytes

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and len(os.path.basename(file_path)) > 6:
            size = os.path.getsize(file_path)
            if size <= min_size:
                print(f"File too small: {filename} ({size} bytes)")
                return False
    return True

def compute_fastq_size(bcl, regex, mult):
    fastq_path = f'/broad/macosko/pipelines/fastqs/{bcl}'
    assert os.path.isdir(fastq_path)
    fastqs = [os.path.join(fastq_path, f) for f in os.listdir(fastq_path)]
    fastqs = [fastq for fastq in fastqs if fastq.endswith(".fastq.gz")]
    fastqs = [fastq for fastq in fastqs if re.compile(regex).search(fastq)]
    assert len(fastqs) >= 2
    fastq_size_bytes = sum(os.path.getsize(fastq) for fastq in fastqs)
    mult_size_gb = round(fastq_size_bytes/1024/1024/1024 * mult)
    mem_size = f"{max(16,mult_size_gb)}G"
    return mem_size

# TODO: add support for variable lanes (--lanes)
# # https://www.10xgenomics.com/support/software/cell-ranger/latest/resources/cr-command-line-arguments
def cellranger_count(bcl, index, transcriptome, chemistry="auto"):
    assert bcl in os.listdir("/broad/macosko/pipelines/fastqs")
    assert transcriptome in os.listdir("/broad/macosko/pipelines/references")
    if os.path.isdir(f'/broad/macosko/pipelines/cellranger-count/{bcl}/{index}'):
        print("Output already exists, run this command:")
        print(f"rm -rf /broad/macosko/data/pipelines/cellranger-count/{bcl}/{index}")
        assert False
    # print(f"BCL: {bcl}")
    # print(f"Index: {index}")
    # print(f"Transcriptome: {transcriptome}")
    # print(f"Chemistry: {chemistry}")
    
    # Assert no whitespace
    for arg in [bcl, index, transcriptome, chemistry]:
        if any(c.isspace() for c in arg):
            print(f"ERROR: Argument '{arg}' contains whitespace")
            sys.exit(1)
    
    ROOT = "/broad/macosko/pipelines"
    BINARY = f"{ROOT}/software/cellranger-8.0.1/bin/cellranger"
    OUTDIR = f"{ROOT}/cellranger-count/{bcl}"
    LOGDIR = f"{ROOT}/logs/{bcl}/{index}"
    
    cellranger_params = (
        f"--id {index} "
        f"--output-dir {OUTDIR}/{index} "
        f"--transcriptome {ROOT}/references/{transcriptome} "
        f"--fastqs {ROOT}/fastqs/{bcl} "
        f"--sample {index} "
        f"--chemistry {chemistry} "
        "--create-bam true "
        "--include-introns true "
        "--nosecondary "
        "--disable-ui"
    )
    
    sbatch_params = (
        f"-C RedHat7 "
        f"-o {LOGDIR}/cellranger-count.log "
        f"-J cellranger-count-{bcl}-{index} "
        f"-c 16 --mem 128G --time 72:00:00 "
        f"--mail-user macosko-pipelines@broadinstitute.org "
        f"--mail-type END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT"
    )
    
    # Ensure output directories exist
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    
    # Build and submit job
    command = f"{BINARY} count {cellranger_params}"
    sbatch_cmd = f"sbatch {sbatch_params} --wrap \"{command}\""
    
    return(sbatch_cmd)

def recon_count_and_run(bcl, index, lane=0, bc1=0, bc2=0, p=1.0, n_neighbors=45, local_connectivity=1, n_epochs=2000):
    assert bcl in os.listdir("/broad/macosko/pipelines/fastqs")
    assert not re.search(r"\s", bcl)
    assert not re.search(r"\s", index)
    assert isinstance(lane, int) and 0 <= lane <= 8 # 0 means all lanes
    assert type(bc1) == type(bc2) == int
    assert 0 < p <= 1
    
    regex = rf"{index}.*" + (rf"_L00{lane}_.*" if lane > 0 else "")
    name = f"{index}" + (f"-{lane}" if lane > 0 else "") + (f"_p-{p}" if p<1 else "") + (f"_bc1-{bc1}" if bc1 > 0 else "") + (f"_bc2-{bc2}" if bc2 > 0 else "")
    in_dir = f"{root}/recon-count/{bcl}/{name}"
    out_dir = f"{root}/recon/{bcl}/{name}"
    log_dir = f"{root}/logs/{bcl}/{name}"
    
    # assert os.path.isdir(in_dir)

    # Get the size of the fastqs
    mem = compute_fastq_size(bcl, regex, 1.5)
    
    # Create the sbatch command
    julia_command = f"julia --threads 8 --heap-size-hint={mem} {root}/scripts/recon-count.jl {root}/fastqs/{bcl} {in_dir} -r {regex} -p {p} -x {bc1} -y {bc2}"
    knn_command = f"micromamba run python {root}/scripts/knn.py -i {in_dir} -o {in_dir} -n 150 -b 2 -c 8 -k 2"
    recon_command = f"micromamba run python {root}/scripts/recon.py -i {in_dir} -o {out_dir} -c 8 -b 2 -nn {n_neighbors} -lc {local_connectivity} -ne {n_epochs}"
    
    command = f"{julia_command} ; {knn_command} ; {recon_command}"
    
    cmd = podman_sbatch_wrapper(command,
                                logpath = f"{log_dir}/recon.log",
                                jobname = f"recon-{bcl}-{name}",
                                mem=mem, cpus=8, time="72:00:00")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return(cmd)

def recon_count(bcl, index, lane=0, bc1=0, bc2=0, p=1.0):
    assert bcl in os.listdir("/broad/macosko/pipelines/fastqs")
    assert not re.search(r"\s", bcl)
    assert not re.search(r"\s", index)
    assert isinstance(lane, int) and 0 <= lane <= 8 # 0 means all lanes
    assert type(bc1) == type(bc2) == int
    assert 0 < p <= 1
    
    regex = rf"{index}.*" + (rf"_L00{lane}_.*" if lane > 0 else "")
    name = f"{index}" + (f"-{lane}" if lane > 0 else "") + (f"_p-{p}" if p<1 else "") + (f"_bc1-{bc1}" if bc1 > 0 else "") + (f"_bc2-{bc2}" if bc2 > 0 else "")
    out_dir = f"{root}/recon-count/{bcl}/{name}"
    log_dir = f"{root}/logs/{bcl}/{name}"
    
    # Get the size of the fastqs
    mem = compute_fastq_size(bcl, regex, 1.5)
    
    # Create the sbatch command
    julia_command = f"julia --threads 8 --heap-size-hint={mem} {root}/scripts/recon-count.jl {root}/fastqs/{bcl} {out_dir} -r {regex} -p {p} -x {bc1} -y {bc2}"
    python_command = f"micromamba run python {root}/scripts/knn.py -i {out_dir} -o {out_dir} -n 150 -b 2 -c 8 -k 2"
    
    command = f"{julia_command} ; {python_command}"
    
    cmd = podman_sbatch_wrapper(command,
                                logpath = f"{log_dir}/recon-count.log",
                                jobname = f"recon-count-{bcl}-{name}",
                                mem=mem, cpus=8, time="24:00:00")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return(cmd)

def recon(bcl, index, lane=0, bc1=0, bc2=0, p=1.0, n_neighbors=45, local_connectivity=1, n_epochs=2000):
    assert bcl in os.listdir("/broad/macosko/pipelines/fastqs")
    assert not re.search(r"\s", bcl)
    assert not re.search(r"\s", index)
    assert isinstance(lane, int) and 0 <= lane <= 8 # 0 means all lanes
    assert type(bc1) == type(bc2) == int
    assert 0 < p <= 1
    
    name = f"{index}" + (f"-{lane}" if lane > 0 else "") + (f"_p-{p}" if p<1 else "") + (f"_bc1-{bc1}" if bc1 > 0 else "") + (f"_bc2-{bc2}" if bc2 > 0 else "")
    in_dir = f"{root}/recon-count/{bcl}/{name}"
    out_dir = f"{root}/recon/{bcl}/{name}"
    log_dir = f"{root}/logs/{bcl}/{name}"
    assert os.path.isdir(in_dir)

    matrix_gb = os.path.getsize(f"{in_dir}/matrix.csv.gz")/1024/1024/1024
    mem = f"{round(max(matrix_gb*25,16))}G"

    python_command = f"micromamba run python {root}/scripts/recon.py -i {in_dir} -o {out_dir} -c 8 -b 2 -nn {n_neighbors} -lc {local_connectivity} -ne {n_epochs}"
    cmd = podman_sbatch_wrapper(python_command,
                                logpath = f"{log_dir}/recon.log",
                                jobname = f"recon-{bcl}-{name}",
                                mem=mem, cpus=8, time="72:00:00")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return(cmd)

# TODO: Multiple pucks
def spatial_count(bcl, index, puck_dir, p=1.0):
    assert bcl in os.listdir("/broad/macosko/pipelines/fastqs")    
    assert not re.search(r"\s", bcl)
    assert not re.search(r"\s", index)
    assert not re.search(r"\s", puck_dir)
    assert 0 < p <= 1
    
    # Get the size of the fastqs
    mem_size = compute_fastq_size(bcl, index, 1.5)

    # Create the sbatch command
    out_dir = f"{index}" + (f"_p-{p}" if p<1 else "")
    julia_command = f"julia --heap-size-hint={mem_size} {root}/scripts/spatial-count.jl {root}/fastqs/{bcl} {puck_dir} {root}/spatial-count/{bcl}/{out_dir} -r {index} -p {p}"
    sbatch_params = f"--nodelist=slurm-bits-d[002-005] -C container -o {root}/logs/{bcl}/{out_dir}/spatial-count.log -J spatial-count-{bcl}-{index} \
                      -c 9 --mem {mem_size} --time 24:00:00 \
                      --mail-user macosko-pipelines@broadinstitute.org --mail-type END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT"
    q="'"
    cmd = f'sbatch {sbatch_params} --wrap "podman run --rm -v {root}:{root} -v /broad/macosko:/broad/macosko:ro pipeline-image {q}{julia_command}{q}"'

    os.makedirs(f"{root}/spatial-count/{bcl}/{out_dir}", exist_ok=True)
    os.makedirs(f"{root}/logs/{bcl}/{out_dir}", exist_ok=True)
    return(cmd)

def positioning(bcl, cellranger_idx, spatial_idx):
    # assert cellranger_idx in os.listdir(f"/broad/macosko/pipelines/cellranger-count/{bcl}")
    # assert spatial_idx in os.listdir(f"/broad/macosko/pipelines/spatial-count/{bcl}")

    # Create the sbatch command
    mem_size = "200G"
    out_dir = f"{spatial_idx}"
    rscript_command = f"Rscript run-positioning.R {root}/cellranger-count/{bcl}/{cellranger_idx} {root}/spatial-count/{bcl}/{spatial_idx} {root}/positioning/{bcl}/{spatial_idx}"
    sbatch_params = f"--nodelist=slurm-bits-d[002-005] -C container -o {root}/logs/{bcl}/{out_dir}/positioning.log -J positioning-{bcl}-{spatial_idx} \
                      -c 9 --mem {mem_size} --time 24:00:00 \
                      --mail-user macosko-pipelines@broadinstitute.org --mail-type END,FAIL,REQUEUE,INVALID_DEPEND,STAGE_OUT,TIME_LIMIT"
    q="'"
    cmd = f'sbatch {sbatch_params} --wrap "podman run --rm -v {root}:{root} -v /broad/macosko:/broad/macosko:ro pipeline-image {q}{rscript_command}{q}"'

    os.makedirs(f"{root}/positioning/{bcl}/{out_dir}", exist_ok=True)
    os.makedirs(f"{root}/logs/{bcl}/{out_dir}", exist_ok=True)
    return(cmd)

def index_parser(string):
    count = Counter(string)
    index_count = count['D'] + count['N']
    assert index_count <= 2

    ret_dict = {}

    ret_dict['index'] = string[0:8] if index_count == 2 else string[0:4]
    ret_dict['lane'] = list(string.split('-')[-1]) if count['-'] == 1 else []

    return ret_dict

def dict2df(indexes):
    rows = []
    assert isinstance(indexes, dict)
    for lane in sorted(indexes.keys()):
        assert lane in [1,2,3,4,5,6,7,8]
        assert isinstance(indexes[lane], list)
        assert all(isinstance(e, str) for e in indexes[lane])
        assert len(indexes[lane])==len(set(indexes[lane])) # no duplicates
        for index in indexes[lane]:
            if bool(re.match(r"^[ND]7\d{2}[ND]5\d{2}$", index)): # [ND]7xx[ND]5yy
                row1 = ND7.loc[ND7['I7_Index_ID'] == index[:4]]
                seq1 = row1["index"].iloc[0]
                row2 = ND5.loc[ND5['I5_Index_ID'] == index[-4:]]
                seq2 = row2["index2_workflow_a"].iloc[0]
            elif index in ND7["I7_Index_ID"].values: # D7xx
                row = ND7.loc[ND7['I7_Index_ID'] == index]
                seq1 = row["index"].iloc[0]
                seq2 = "CGAGATCT"
            elif index in TT["index_name"].values: # SI-TT-xx
                row = TT.loc[TT['index_name'] == index]
                seq1 = row["index(i7)"].iloc[0]
                seq2 = row["index2_workflow_a(i5)"].iloc[0]
            elif index in NT["index_name"].values: # SI-NT-xx
                row = NT.loc[NT['index_name'] == index]
                seq1 = row["index(i7)"].iloc[0]
                seq2 = row["index2_workflow_a(i5)"].iloc[0]
            elif index in NN["index_name"].values: # SI-NN-xx
                row = NN.loc[NN['index_name'] == index]
                seq1 = row["index(i7)"].iloc[0]
                seq2 = row["index2_workflow_a(i5)"].iloc[0]
            elif index in TS["index_name"].values: # SI-TS-xx
                row = TS.loc[TS['index_name'] == index]
                seq1 = row["index(i7)"].iloc[0]
                seq2 = row["index2_workflow_a(i5)"].iloc[0]
            else:
                raise IndexError(f"ERROR: index {index} not found")

            rows.append([lane, index, seq1, seq2])
    
    df = pd.DataFrame(rows, columns=['Lane', 'Sample_ID', 'index', 'index2'])

    # add padding to 8bp indexes if some indexes are 10bp
    if any(df['index'].str.len() == 10) or any(df['index2'].str.len() == 10):
        df['index'] = df['index'].apply(lambda x: x+'AT' if len(x) == 8 else x)
        df['index2'] = df['index2'].apply(lambda x: 'AC'+x if len(x) == 8 else x)

    # bcl-convert requires all indexes to have the same length
    assert df['index'].str.len().nunique() == 1
    assert df['index2'].str.len().nunique() == 1
    
    return df
