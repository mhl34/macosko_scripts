#!/bin/bash

# packages:
# needs cell_type_mapper installed

# inputs:
# annotated h5ad path
# query h5ad path
# output path

# Process the flags and arguments
while [[ "$#" -gt 0 ]]; do
	case $1 in
        	-r|--ref) # For flags for reference h5ad
	        	ref="$2"
			shift # Shift past the argument value
			;;
		-q|--query) # For flags for query h5ad
			query="$2"
			shift
			;;
		-a|--ref_name) # 
			ref_name="$2"
			shift
			;;
		-b|--query_name)
			query_name="$2"
			shift
			;;
		-h|--help) # Help flag
			echo "Usage: $0 [-v|--verbose] [-f|--file filename] [-h|--help]"
			exit 0
			;;
		*) # Unknown option
			echo "Unknown option: $1"
			exit 1
			;;
	esac
	shift # Shift to the next argument
done

if [ -z "$ref" ]; then
	echo "Error: The -r or --ref flag option is required."
	exit 1
fi
if [ -z "$query" ]; then
	echo "Error: The -q or --query flag option is required."
	exit 1
fi
if [ -z "$ref_name" ]; then
	echo "Error: The -a or --ref_name flag option is required."
	exit 1
fi
if [ -x "$query_name" ]; then
	echo "Error: The -b or --query_name flag option is required."
fi

echo "python"
# python create_seurat_obj.py --ref $ref --query $query --ref_name $ref_name --query_name $query_name

ref_high_dir=`dirname ref`
query_high_dir=`dirname query`

ref_dir="$ref_high_dir/$ref_name"
query_dir="$query_high_dir/$query_name"

echo "$ref_dir"
echo "$query_dir"

echo "R"
Rscript --vanilla cs_integration_seurat.R $ref_dir $query_dir $ref_name $query_name
