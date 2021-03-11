import os
from src.utils.helper import predict_on_generator
from src.data_loader.RNASeqLoader import RNASeqDataGenerator
from scipy.stats import spearmanr, pearsonr
from scipy.stats.kde import gaussian_kde
from matplotlib import colors
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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