import numpy as np
from tqdm import tqdm

# predict on generator, return predictions and labels
def predict_on_generator(model, generator):
    preds = []
    labels = []
    pbar = tqdm(total=len(generator))
    for i in range(len(generator)):
        pbar.update(1)
        x, y, _ = generator[i]
        p = model.predict(x, batch_size=x.shape[0], use_multiprocessing=True, workers=20)
        preds.append(p)
        labels.append(y)
    pbar.close()
    preds = np.concatenate(preds)
    preds = preds.astype('float64') 
    labels = np.concatenate(labels)
    labels = labels.astype('float64')
    return preds, labels