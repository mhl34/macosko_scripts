import gspread
import re
import numpy as np
import argparse
import os
import xml.etree.ElementTree as ET
from script_helpers import *

def get_args():
    parser = argparse.ArgumentParser(description='run recon pipeline')
    parser.add_argument("-w", "--walkup", help="WALKUP ID", type=str, default=".")
    parser.add_argument("-bc", "--bcl", help="", type=str)
    parser.add_argument("-l", "--lab", type=str, default="BICAN")
    args, unknown = parser.parse_known_args()
    [print(f"WARNING: unknown command-line argument {u}") for u in unknown]
    return args

args = get_args()
walkup = args.walkup             ; print(f"walkup: {walkup}")
bcl = args.bcl                   ; print(f"bcl: {bcl}")
lab = args.lab                   ; print(f"lab: {lab}")

gspread_client = gspread.service_account(filename="/broad/macosko/leematth/scripts/new_sheets_key.json")
spreadsheet = gspread_client.open("Demux Log")
sheet = spreadsheet.worksheet('Runs')

# index IDs
# Lab    Project	Walkup ID	BCL	   Indexes ([ND]7xx[ND]5yy-lane)
lab_idx = 1
project_idx = 2
walkup_idx = 3
bcl_idx = 4
index_name_idx = 5
lane_idx = 6

# find the project row
output_row = sheet.find(bcl, in_column = bcl_idx).row
indexes_names = sheet.cell(output_row, index_name_idx).value.split(",")
lab = sheet.cell(output_row, lab_idx).value
num_lanes = int(sheet.cell(output_row, lane_idx).value)

if lab == 'Macosko':
    bcl = '/broad/macosko_storage/macosko_lab_GP_depo/' + bcl
else:
    bcl = '/broad/gpbican/mccarroll_bican_bcls/' + bcl

indexes = {i+1: indexes_names for i in range(8)}
samplesheet = dict2df(indexes)
tree = ET.parse(os.path.join(bcl, 'RunInfo.xml'))  # Replace 'data.xml' with your file path
root = tree.getroot()
cycles = [int(item.get("NumCycles")) for item in root.findall('Run/Reads/Read')]

assert len(samplesheet) == len(samplesheet.drop_duplicates())
sheet_path = os.path.join("/broad/macosko/pipelines/samplesheets", basename(bcl), "SampleSheet.csv")
print(sheet_path)

# Write the header
os.makedirs(os.path.dirname(sheet_path), exist_ok=True)
with open(sheet_path, 'w') as f:
    f.write("[Settings]\n")
    f.write("CreateFastqForIndexReads,0\n") # default: 0
    f.write("NoLaneSplitting,false\n") # default: false
    # f.write("BarcodeMismatchesIndex1,1\n") # default: 1
    # f.write("BarcodeMismatchesIndex2,1\n") # default: 1
    assert type(cycles) == list and all(type(c) == int for c in cycles)
    if len(cycles) == 3:
        i1len = len(samplesheet["index"][0])
        assert cycles[0] > 0 and cycles[1] >= i1len and cycles[2] > 0
        R1 = f"Y{cycles[0]}"
        R2 = f"I{i1len}" + ("" if cycles[1]==i1len else f"N{cycles[1]-i1len}")
        R3 = f"Y{cycles[2]}"
        f.write(f"OverrideCycles,{R1};{R2};{R3}\n")
    elif samplesheet['index2'].nunique() == 1:
        i1len = len(samplesheet["index"][0])
        assert cycles[0] > 0 and cycles[1] >= i1len and cycles[2] > 0 and cycles[3] > 0
        R1 = f"Y{cycles[0]}"
        R2 = f"I{i1len}" + ("" if cycles[1]==i1len else f"N{cycles[1]-i1len}")
        R3 = f"N{cycles[2]}"
        R4 = f"Y{cycles[3]}"
        f.write(f"OverrideCycles,{R1};{R2};{R3};{R4}\n")
    elif len(cycles) == 4:
        i1len = len(samplesheet["index"][0])
        i2len = len(samplesheet["index2"][0])
        assert cycles[0] > 0 and cycles[1] >= i1len and cycles[2] >= i2len and cycles[3] > 0
        R1 = f"Y{cycles[0]}"
        R2 = f"I{i1len}" + ("" if cycles[1]==i1len else f"N{cycles[1]-i1len}")
        R3 = ("" if cycles[2]==i2len else f"N{cycles[2]-i2len}") + f"I{i2len}"
        R4 = f"Y{cycles[3]}"
        f.write(f"OverrideCycles,{R1};{R2};{R3};{R4}\n")
    else:
        assert False
    f.write("\n[Data]\n")

# barcode_dict = {x: y for x,y in zip(indexes_names, barcode1_names)}
# samplesheet_dict = {x:y for x,y in zip(samplesheet['Sample_ID'], samplesheet['index'])}
# revcomp_bool = barcode_dict != samplesheet_dict
# revcomp_bool = True
revcomp_bool = False

if num_lanes < 8:
    samplesheet = samplesheet[samplesheet['Lane'] <= num_lanes]

# Write the indexes
if len(cycles) == 3 or samplesheet['index2'].nunique() == 1:
    if revcomp_bool:
        samplesheet['index'] = samplesheet['index'].apply(lambda x: x.translate(str.maketrans('ACGTacgt', 'TGCAtgca'))[::-1])
    samplesheet[['Lane', 'Sample_ID', 'index']].to_csv(sheet_path, mode='a', index=False)
elif len(cycles) == 4:
    if revcomp_bool:
        samplesheet['index'] = samplesheet['index'].apply(lambda x: x.translate(str.maketrans('ACGTacgt', 'TGCAtgca'))[::-1])
    samplesheet.to_csv(sheet_path, mode='a', index=False)
else:
    assert False

print(f"/broad/macosko/pipelines/scripts/bcl-convert.sh {bcl}")
os.system(f"/broad/macosko/pipelines/scripts/bcl-convert.sh {bcl}")
