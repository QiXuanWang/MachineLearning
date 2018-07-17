import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from data import gen_data, save_plot, normalize, denormalize
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
        MaxPooling1D(padding='same'),     # Downsample the output of convolution by 2X.
        BatchNormalization(epsilon=1e-5),
        Conv1D(filters=int(nb_filter/2), kernel_size=filter_length, padding='same', activation='relu'),
        MaxPooling1D(padding='same'), # seq_len/4
        BatchNormalization(epsilon=1e-5),

        # add another layer but not helpful 
        # results: loss=1.5e-5, val_loss=2.2e-5
        #Conv1D(filters=int(nb_filter/2), kernel_size=filter_length, padding='same', activation='relu'),
        #AveragePooling1D(padding='same'), # seq_len/4
        #BatchNormalization(epsilon=1e-5),
        #UpSampling1D(),
        #Conv1D(filters=int(nb_filter/2), kernel_size=filter_length, padding='same', activation='relu'),
        #BatchNormalization(epsilon=1e-5),
        ## 3rd layer end

        UpSampling1D(),
        Conv1D(filters=int(nb_filter/2), kernel_size=filter_length, padding='same', activation='relu'),
        BatchNormalization(epsilon=1e-5),
        UpSampling1D(),
        Conv1D(filters=nb_filter, kernel_size=filter_length, padding='same', activation='relu'),
        BatchNormalization(epsilon=1e-5),

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

    X = np.atleast_3d(timeseries)
    y = np.atleast_3d(labels)
    print('\nShape: {}: y:{}'.format(X.shape, y.shape))

    test_size = int(0.1*len(timeseries))
    train_size = int(0.9*len(timeseries))
    #X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    idx = np.random.choice(len(timeseries), len(timeseries), replace=False)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    global eval_data_scale, eval_label_scale
    eval_data_scale = data_scale[test_idx, :]
    eval_label_scale = label_scale[test_idx, :]
    X_train, X_test, y_train, y_test = X[train_idx, :], X[test_idx, :], y[train_idx, :], y[test_idx, :]

    #train
    model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))
    model.save_weights('try5_keras4.hd5')

    pred = model.predict(X_test)
    X_test = denormalize(X_test, eval_data_scale, axis=1)
    y_test = denormalize(y_test, eval_label_scale, axis=1)
    pred = denormalize(pred, eval_label_scale, axis=1)
    save_plot(X_test, y_test, pred, 'try5_keras4.out', style='keras')


def main():
    #_data, _label = gen_data(filename='1cycle_iv_small.txt',input_list=['v(i)'], output_list=['v(pad)'])
    _data, _label = gen_data(filename='1cycle_iv_small_100p.txt',input_list=['v(i)'], output_list=['i(vi)'])
    #_data, _label = gen_data(filename='1cycle_iv_small_10p.txt',input_list=['v(i)'], output_list=['i(vpad)'])
    _data = np.concatenate([_data]*60)
    _label = np.concatenate([_label]*60)
    global data_scale, label_scale
    data_scale = normalize(_data, axis=1)
    label_scale = normalize(_label, axis=1)

    window_size = len(_data[0]) # actually it's sequence length or num_steps
    evaluate_timeseries_with_label(_data, _label, window_size)

if __name__ == "__main__":
    main()
