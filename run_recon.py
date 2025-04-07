import gspread
import re
import argparse
import os
from script_helpers import *

def get_args():
    parser = argparse.ArgumentParser(description='run recon pipeline')
    parser.add_argument("-p", "--project", help="name of project (i.e. BICAN, Stress Atlas, etc.)", type=str, default=".")
    parser.add_argument("-n", "--name", help="", type=str, default=".")
    parser.add_argument("-bc", "--bcl", help="", type=str)
    parser.add_argument("-c", "--cores", type=int, default=-1)
    parser.add_argument("-b", "--bead", type=int, default=2)
    parser.add_argument("-rt", "--runtype", type=str, default="Runs")
    
    args, unknown = parser.parse_known_args()
    [print(f"WARNING: unknown command-line argument {u}") for u in unknown]
    return args

args = get_args()

project = args.project           ; print(f"project: {project}")
name = args.name                 ; print(f"name: {name}")
bcl = args.bcl                   ; print(f"bcl: {bcl}")
cores = args.cores               ; print(f"cores: {cores}")
bead = args.bead                 ; print(f"bead: {bead}")
runtype = args.runtype           ; print(f"run type: {runtype}")


# get key and open sheet
gspread_client = gspread.service_account(filename="/broad/macosko/leematth/scripts/new_sheets_key.json")
spreadsheet = gspread_client.open("Recon Log")

sheet = spreadsheet.worksheet(runtype)

commands = []

if runtype == 'Runs':
    # hard_code column indices
    project_col_idx = 1
    name_col_idx = 2
    bcl_col_idx = 3
    indexes_col_idx = 4
    num_lanes = 8
    
    # find the project row
    output_row = sheet.find(bcl, in_column = bcl_col_idx).row
    index_list = sheet.cell(output_row, indexes_col_idx).value.split(",")
    index_dicts = [index_parser(index_string) for index_string in index_list]
    print('Generating commands...')
    for index_dict in index_dicts:
        lanes = index_dict['lane']
        indexes = index_dict['index']
        if len(lanes) == 0:
            commands.extend([recon_count_and_run(bcl, indexes)])
        else:
            commands.extend([recon_count_and_run(bcl, indexes, lane = l) for l in lanes])

if runtype == 'Re-Runs':
    # hard_code column index
    project_col_idx = 1
    name_col_idx = 2
    bcl_col_idx = 3
    indexes_col_idx = 4
    bc1_idx = 5
    bc2_idx = 6
    p_idx = 7
    num_lanes = 8
    
    # find the project row
    output_row = sheet.find(bcl, in_column = bcl_col_idx).row
    index_list = sheet.cell(output_row, indexes_col_idx).value.split(",")
    bc1 = int(sheet.cell(output_row, bc1_idx).value)
    bc2 = int(sheet.cell(output_row, bc2_idx).value)
    p = float(sheet.cell(output_row, p_idx).value)
    index_dicts = [index_parser(index_string) for index_string in index_list]
    print('Generating commands...')
    for index_dict in index_dicts:
        lanes = index_dict['lane']
        indexes = index_dict['index']
        if len(lanes) == 0:
            commands.extend([recon_count_and_run(bcl, indexes, bc1=bc1, bc2=bc2, p=p)])
        else:
            commands.extend([recon_count_and_run(bcl, indexes, lane=l, bc1=bc1, bc2=bc2, p=p) for l in lanes])

print('Running commands...')
for command in commands:
    print(command)
    os.system(command)
