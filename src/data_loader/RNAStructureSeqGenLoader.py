import os
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import keras
from keras.utils import Sequence
from .combine_letter_profiles import getCombinedRNAStructure as gCRS
import subprocess
from random import randint
from multiprocessing.pool import ThreadPool
def seq_to_onehot(seq): 
    seq = seq.strip()
    values = list(seq+"ACGT") 
    label_encoder = LabelEncoder();
    integer_encoded = label_encoder.fit_transform(values);
    onehot_encoder = OneHotEncoder(sparse=False);
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1);
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded); 
    return onehot_encoded[:-4]
def getEditData(line):
    data = line.decode("latin-1").split("_")
    return float(data[3])

def writeFasta(sequences, location):
    i = 1
    with open(location, "w") as f:
        for seq in sequences:
            f.write('>seq {}\n'.format(i))
            i+=1
            f.write(seq+'\n')    
def runCommand(command):
    subprocess.run(command, shell=True)
def getBatchStructureData(sequences):
    hsh = str(randint(0, 9999999999))
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'tmp/rna_structure_temp-{}.fasta'.format(hsh))
    writeFasta(sequences, filename)
    
    pool = ThreadPool(processes=4)
    commands = list()
    for x in ["E", "M", "H", "I"]:
        exe_path = os.path.join(dirname, '{}_RNAplfold'.format(x))
        out_path = os.path.join(dirname, 'tmp/{}_profile-{}.txt'.format(x, hsh))
        command = "{} -W 240 -L 160 -u 1 <{} >{}".format(exe_path, filename, out_path)
        commands.append(command)
    pool.map(runCommand, commands)
    ret = gCRS(os.path.join(dirname, 'tmp/E_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/H_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/I_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/M_profile-{}.txt'.format(hsh)))
    subprocess.run("rm {}".format(filename), shell=True)
    subprocess.run("rm {} {} {} {}".format(os.path.join(dirname, 'tmp/E_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/H_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/I_profile-{}.txt'.format(hsh)), os.path.join(dirname, 'tmp/M_profile-{}.txt'.format(hsh))), shell=True)
    return ret
class RNASeqStructDataGenerator(keras.utils.Sequence):
    
    def __init__(self, h5_filepath, batch_size, n_around_center=50):
        self.file_name = h5_filepath
        self.batch_size = batch_size
        with h5py.File(self.file_name, "r") as f:
            self.elements = f['sequences'].shape[0]
            self.dim = 2*n_around_center+1
            self.f_idx = len(f['sequences'][0]) // 2 - n_around_center
            self.e_idx = len(f['sequences'][0]) // 2 + n_around_center + 1
            self.n_channels = 4
        self.indexes = np.arange(self.elements)
        
    def __len__(self):
        return self.elements // self.batch_size
    
    def __on_epoch_end__(self):
        self.indexes = np.arange(self.elements)
        np.random.shuffle(self.indexes)
        
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]     
        return self.__data_generation(indexes)
    
    def __data_generation(self, list_IDs_temp): 
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty((self.batch_size))  
        seq = list()
        with h5py.File(self.file_name, "r") as f: 
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = seq_to_onehot(f['sequences'][ID].decode("utf-8")[self.f_idx:self.e_idx])
                seq.append(f['sequences'][ID].decode("utf-8")[self.f_idx:self.e_idx].strip())
                y[i] = getEditData(f['metadata'][ID])
        #rna_structure_data = getBatchStructureData(seq)[:,self.f_idx:self.e_idx, :]
        rna_structure_data = getBatchStructureData(seq)
        X = np.concatenate((X, rna_structure_data), axis=2)
        return X, y