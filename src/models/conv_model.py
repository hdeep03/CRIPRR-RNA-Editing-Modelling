from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Flatten, BatchNormalization, MaxPool1D, Activation
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import backend as K


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r


def build_model(seq_length, dim, filters, kernel_size, dense_net, blocks=2, dil_rate=1, dropout=0.0, pooling_size=1):
    model = Sequential()
    model.add(Input(shape=(seq_length,dim)))
    
    for x in range(blocks):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate = dil_rate, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(dropout))
        if pooling_size!=1:
            model.add(MaxPool1D(pool_size=pooling_size, padding="same"))
                       
    model.add(Flatten())
    for x in dense_net:
        model.add(Dense(x, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
                  
    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse', correlation_coefficient_loss, pearson_r])
    return model