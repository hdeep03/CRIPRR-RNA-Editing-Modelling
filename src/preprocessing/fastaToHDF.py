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

def seq_to_onehot(seq): 
    seq = seq.strip()
    values = list(seq) 
    label_encoder = LabelEncoder();
    integer_encoded = label_encoder.fit_transform(values);
    onehot_encoder = OneHotEncoder(sparse=False);
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1);
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded); 
    return onehot_encoded 
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
        f.create_dataset("sequences", data=sequences)
        f.create_dataset("metadata", data=meta)
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError('arg1: fasta file path; arg2: output file path(Without the extension)')
    fasta_path = sys.argv[1]
    output_path = sys.argv[2]
    main(fasta_path, output_path)
    