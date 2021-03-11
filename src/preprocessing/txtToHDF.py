# description: convert fastas to HDFs

import os
import re
import sys
import gzip
import glob
import h5py
import logging
import subprocess
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm

def main(fasta_path, output_path):
    print(output_path)
    print(fasta_path)
    seq = list()
    metadata = list()
    if fasta_path.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
    with opener(fasta_path,'rt') as f:
        for line in tqdm(f):
            if line[0:1] == ">":
                y = line[1:].strip().encode("utf-8")
                metadata.append(y)
            else:
                x = line.strip().encode("utf-8")
                seq.append(x)
    assert len(seq)==len(metadata)
    sequences = np.stack(seq)
    meta = np.stack(metadata)
    with h5py.File("{}.hdf5".format(output_path), "w") as f:
        f.create_dataset("structure", data=sequences)
        f.create_dataset("metadata", data=meta)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError('arg1: fasta file path; arg2: output file path(Without the extension)')
    fasta_path = sys.argv[1]
    output_path = sys.argv[2]
    main(fasta_path, output_path)
    