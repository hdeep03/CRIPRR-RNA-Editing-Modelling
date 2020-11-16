from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, Flatten, BatchNormalization, MaxPool1D
from tensorflow.keras.optimizers import RMSprop

def build_model(seq_length,dim, filters, kernel_size, blocks=2, dil_rate=1, dropout=0.0):
    model = Sequential()
    model.add(Input(shape=(seq_length,dim)))
    
    for x in range(blocks):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate = dil_rate, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(MaxPool1D(pool_size=2, padding="same"))
                       
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
                  
    optimizer = RMSprop()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model