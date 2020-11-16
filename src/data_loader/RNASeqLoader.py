import os
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import keras
from keras.utils import Sequence

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

class RNASeqDataGenerator(keras.utils.Sequence):
    
    def __init__(self, h5_filepath, batch_size, reverse=True, n_around_center=50):
        self.file_name = h5_filepath
        self.batch_size = batch_size
        self.reverse = reverse
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
        if self.reverse:
            X = np.empty((2*self.batch_size, self.dim, self.n_channels))
            y = np.empty((2*self.batch_size))      
        with h5py.File(self.file_name, "r") as f: 
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = seq_to_onehot(f['sequences'][ID].decode("utf-8")[self.f_idx:self.e_idx])
                y[i] = getEditData(f['metadata'][ID])
                if self.reverse:
                    X[self.batch_size+i,] = np.flipud(X[i])
                    y[self.batch_size+i] = y[i]
        return X, y