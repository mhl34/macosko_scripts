import anndata
import argparse

class CS_Integration:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'files',
            metavar='F',
            type=str,
            nargs='+',
            help='an input file'
        )
        args = parser.parse_args()