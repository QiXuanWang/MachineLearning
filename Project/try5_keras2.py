import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from data import gen_data, save_plot
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import BatchNormalization, UpSampling1D, MaxPooling1D, Dense, AveragePooling1D
from keras import callbacks
callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5)

def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
    # conv1d input:
    # (batch_size, steps, input_dim) ==> steps: sequence_len, input_dim: vector_len
    # conv1d output:  (nb_filter, vector_len)
    #  (batch_size, new_steps, filters) ==>  new_steps < sequence_len , filters: any value
    # filter_length: it's internal parameter
    # nd_outputs: == nb_inputs == vector_len?
    model = Sequential((
        Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same', activation='relu', input_shape=(window_size, nb_input_series)),
        AveragePooling1D(padding='same'),     # Downsample the output of convolution by 2X.
        BatchNormalization(epsilon=1e-8),
        Conv1D(filters=int(nb_filter/2), kernel_size=filter_length, padding='same', activation='relu'),
        AveragePooling1D(padding='same'), # seq_len/4
        BatchNormalization(epsilon=1e-8),

        UpSampling1D(),
        Conv1D(filters=int(nb_filter/2), kernel_size=filter_length, padding='same', activation='relu'),
        BatchNormalization(epsilon=1e-8),
        UpSampling1D(),
        Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same', activation='relu'),
        BatchNormalization(epsilon=1e-8),

        Dense(nb_outputs, activation='linear'),     # For binary classification, change the activation to 'sigmoid'
        # output: nb_filter*seq_len/4, nb_outputs
    ))
    #model.compile(loss='mse', optimizer='sgd', metrics=['mae', 'acc'])
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # To perform (binary) classification instead:
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model

def evaluate_timeseries_with_label(timeseries, labels, window_size):
    filter_length = 128
    nb_filter = 64
    nb_series = 1
    nb_samples = timeseries.shape[0]

    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series))
    model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=nb_series, nb_outputs=nb_series, nb_filter=nb_filter)
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
    model.summary()

    test_size = int(0.2 * len(timeseries))
    X = np.atleast_3d(timeseries)
    y = np.atleast_3d(labels)
    print('\nShape: {}: y:{}'.format(X.shape, y.shape))

    #X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    X_train, X_test, y_train, y_test = X[:], X[-test_size:], y[:], y[-test_size:]
    model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))
    model.save_weights('try5_keras2.hd5')
    # to load
    #model.load_weights('try5_keras2.hd5')

    pred = model.predict(X_test)
    save_plot(X_test, y_test, pred, 'try5_keras2.out', style='keras')
    #save(X_test, y_test, pred)


def save(X, y, predicted):
    print("Original shape:")
    print(X.shape)
    print(y.shape)
    print(predicted.shape)
    index = np.random.randint(1, X.shape[0], 20)
    #X = X.squeeze()[:10]
    #y = y.squeeze()[:10]
    #predicted = predicted.squeeze()[:10]
    X = X.squeeze()[index]
    y = y.squeeze()[index]
    predicted = predicted.squeeze()[index]
    print("after:")
    print(X.shape)
    print(y.shape)
    print(predicted.shape)
    d={'x':X, 'y': y, 'p':predicted}
    df = pd.DataFrame().append(d, ignore_index=True)
    df.to_pickle('try5_keras2.out')


def main():
    # no meaningful results
    #_data, _label = gen_data(filename='500cycle.txt',input_list=['v(i)'], output_list=['v(pad)'], seq_len=8) # multiple cycle won't help at all
    #_data, _label = gen_data(filename='1cycle_short.txt',input_list=['v(i)'], output_list=['v(pad)']) 
    #_data, _label = gen_data(filename='1cycle_iv_2.txt',input_list=['v(i)'], output_list=['v(pad)'])  # rload not good
    _data, _label = gen_data(filename='1cycle_iv.txt',input_list=['v(i)'], output_list=['i(vpad)']) 
    _data = np.concatenate([_data]*10) # duplicate
    _label = np.concatenate([_label]*10)
    window_size = len(_data[0]) # actually it's sequence length or num_steps
    evaluate_timeseries_with_label(_data, _label, window_size)

if __name__ == "__main__":
    main()
