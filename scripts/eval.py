import os
import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import keras
from keras.utils import Sequence
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

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
import tensorflow as tf
import sys
import os

class Evaluator():
    def __init__(self, model, test_data_loader):
        self.model = model
        self.test_data_loader = test_data_loader
    def evaluate(self):
        self.preds, self.labels = predict_on_generator(self.model, self.test_data_loader)
        self.preds = self.preds.flatten()
        self.labels = self.labels.flatten()
        spearman_r, spearman_p_value = spearmanr(self.labels, self.preds)
        pearson_r, pearson_p_value = pearsonr(self.labels, self.preds)
        print('Spearman Correlation : {:.4f}'.format(spearman_r))
        print('Pearson Correlation  : {:.4f}'.format(pearson_r))
        
        
    def generatePlot(self, title):
        nbins = 100
        x = self.preds.flatten()
        y = self.labels.flatten()
        k = gaussian_kde([x,y])
        xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        plt.style.use(['dark_background'])
        exp = "158B Chrom 20-X"
        plt.axis('square')
        plt.ylabel("Observed from Experiment {}".format(exp))
        plt.xlabel("Predicted from Experiment {}".format(exp))
        plt.title(title)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape),cmap=plt.cm.inferno, norm=colors.LogNorm(vmin=1e-2), snap=False)
        plt.show()
if __name__ == "__main__":
    os. chdir("/home/harsh")
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    data_loader = RNASeqDataGenerator(data_path, 1024)
    model = tf.keras.models.load_model(model_path)
    e = Evaluator(model, data_loader)
    e.evaluate()
    