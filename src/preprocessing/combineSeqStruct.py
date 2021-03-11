# Utility to combine the [train/validation/test]_data.hdf5 file and [train/validation/test]_data.RNA.hdf5 file 

import h5py
import os
import numpy as np
import sys

if __name__=="__main__":
    rnadata=None
    with h5py.File(sys.argv[2], "r") as hf:
        rnadata=hf['structures'][()]
    with h5py.File(sys.argv[1], "a") as hf:
        hf.create_dataset("structure", data=rnadata)