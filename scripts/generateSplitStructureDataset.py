import json
import h5py
import os
import sys
import numpy as np

def parseConfigFile(json_path):
    data = json.load(open(json_path))
    data_dir = data['data_dir']
    target_dir = data['target_dir']
    training_files = [os.path.join(data_dir, x) for x in data['train_files']]
    validation_files = [os.path.join(data_dir, x) for x in data['validation_files']]
    testing_files = [os.path.join(data_dir, x) for x in data['test_files']]
    return target_dir, training_files, validation_files, testing_files
def buildDatasetFromShards(output_path, files):
    seq = list()
    meta = list()
    for file in files:
        with h5py.File(file,'r') as f:
            seq.append(f['structure'][()])
            meta.append(f['metadata'][()])
    sequences = np.concatenate(seq, axis=0)
    metadata = np.concatenate(meta, axis=0)
    print("{} structures in {}".format(sequences.shape[0], output_path))
    with h5py.File("{}.hdf5".format(output_path),'w') as f:
        f.create_dataset("structures", data=sequences)
        f.create_dataset("metadata", data=metadata)
        
def main(config_path):
    data_dir, training_files, validation_files, testing_files = parseConfigFile(config_path)
    buildDatasetFromShards(os.path.join(data_dir, "train_structure_data"), training_files)
    buildDatasetFromShards(os.path.join(data_dir, "validation_structure_data"), validation_files)
    buildDatasetFromShards(os.path.join(data_dir, "test_structure_data"), testing_files)      
if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError('Only arg is json file path')
    config_path = sys.argv[1]
    main(config_path)
    