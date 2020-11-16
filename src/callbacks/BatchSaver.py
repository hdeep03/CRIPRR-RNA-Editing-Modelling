import tensorflow as tf
class BatchSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, fpath):
        self.filepath = fpath
    def on_batch_end(self, batch,logs={}):
        if batch%100==0:
            self.model.save_weights(self.filepath+"Batch-{}.hdf5".format(batch))