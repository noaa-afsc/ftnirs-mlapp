import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import pyCompare 
import matplotlib
import keras_tuner as kt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from yellowbrick.regressor import PredictionError, ResidualsPlot
from matplotlib import pyplot as plt
np.random.seed(42)
tf.random.set_seed(42)
from collections import Counter
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras_tuner.tuners import BayesianOptimization, Hyperband
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
import pywt
from sklearn.decomposition import PCA
import sys

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sns.set(rc = {'figure.figsize': (10,6)})
sns.set(style="whitegrid",font_scale = 2 )

# # Read and Clean Data
def read_and_clean_data(filepath, drop_outliers=True):
    data = pd.read_csv(filepath)

    if drop_outliers:
        data.drop(data[data['sample'] == 'outlier'].index, inplace=True)
    
    data.rename(columns={"final_age": "Age"}, inplace=True)
    if data.isnull().values.any():
        raise ValueError("Data contains NaN values")
    
    X_train = data[data['sample'] == 'training']
    X_test = data[data['sample'] == 'test']
    
    y_train = X_train.pop('Age')
    y_test = X_test.pop('Age')
    f_train = X_train.pop('file_name')
    f_test = X_test.pop('file_name')
    s_train = X_train.pop('sample')
    s_test = X_test.pop('sample')
    
    X_train_A = X_train[X_train.columns[0:4]]
    X_test_A = X_test[X_test.columns[0:4]]
    X_train_B = X_train[X_train.columns[4:]]
    X_test_B = X_test[X_test.columns[4:]]
    
    return data, X_train, X_test, y_train, y_test, f_train, f_test, s_train, s_test, X_train_A, X_test_A, X_train_B, X_test_B


# Savitzky-Golay filter function
def savgol_filter_func(data, window_length=17, polyorder=2, deriv=1):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, deriv=deriv)

# Moving Average filter function
def moving_average_filter(data, size=5):
    return uniform_filter1d(data, size=size, axis=1)

# Gaussian filter function
def gaussian_filter_func(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma, axis=1)

# Median filter function
def median_filter_func(data, size=5):
    return median_filter(data, size=(1, size))

# Wavelet filter function
def wavelet_filter_func(data, wavelet='db1', level=1):
    def apply_wavelet(signal):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        coeffs[1:] = [pywt.threshold(i, value=0.5 * max(i)) for i in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet)
    
    return np.apply_along_axis(apply_wavelet, axis=1, arr=data)

# Fourier filter function
def fourier_filter_func(data, threshold=0.1):
    def apply_fft(signal):
        fft_data = np.fft.fft(signal)
        frequencies = np.fft.fftfreq(len(signal))
        fft_data[np.abs(frequencies) > threshold] = 0
        return np.fft.ifft(fft_data).real
    
    return np.apply_along_axis(apply_fft, axis=1, arr=data)

# PCA filter function
def pca_filter_func(data, n_components=5):
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)
    return pca.inverse_transform(transformed)

# Main preprocessing function
def preprocess_spectra(X_train_B, X_test_B, filter_type='savgol'):
    filter_functions = {
        'savgol': savgol_filter_func,
        'moving_average': moving_average_filter,
        'gaussian': gaussian_filter_func,
        'median': median_filter_func,
        'wavelet': wavelet_filter_func,
        'fourier': fourier_filter_func,
        'pca': pca_filter_func
    }
    
    filter_func = filter_functions.get(filter_type, savgol_filter_func)
    
    preprTR = filter_func(X_train_B.values)
    preprTE = filter_func(X_test_B.values)
    
    X_train_B = pd.DataFrame(preprTR, index=X_train_B.index, columns=X_train_B.columns)
    X_test_B = pd.DataFrame(preprTE, index=X_test_B.index, columns=X_test_B.columns)
    
    return X_train_B, X_test_B

def apply_normalization(X_train_A, X_test_A):
    normalizer = Normalizer()
    X_train_A = pd.DataFrame(normalizer.transform(X_train_A), columns=X_train_A.columns)
    X_test_A = pd.DataFrame(normalizer.transform(X_test_A), columns=X_test_A.columns)
    
    return X_train_A, X_test_A


def apply_robust_scaling(X_train_A, X_test_A, y_train, y_test):
    scaler_y = RobustScaler()
    scaler_y.fit(y_train.values.reshape(-1, 1))
    y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    scaler_x = RobustScaler()
    scaler_x.fit(X_train_A)
    X_train_A = pd.DataFrame(scaler_x.transform(X_train_A), columns=X_train_A.columns)
    X_test_A = pd.DataFrame(scaler_x.transform(X_test_A), columns=X_test_A.columns)
    
    return X_train_A, X_test_A, y_train, y_test, scaler_y


def apply_minmax_scaling(X_train_A, X_test_A, y_train, y_test):
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.values.reshape(-1, 1))
    y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train_A)
    X_train_A = pd.DataFrame(scaler_x.transform(X_train_A), columns=X_train_A.columns)
    X_test_A = pd.DataFrame(scaler_x.transform(X_test_A), columns=X_test_A.columns)
    
    return X_train_A, X_test_A, y_train, y_test, scaler_y


def apply_maxabs_scaling(X_train_A, X_test_A, y_train, y_test):
    scaler_y = MaxAbsScaler()
    scaler_y.fit(y_train.values.reshape(-1, 1))
    y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    scaler_x = MaxAbsScaler()
    scaler_x.fit(X_train_A)
    X_train_A = pd.DataFrame(scaler_x.transform(X_train_A), columns=X_train_A.columns)
    X_test_A = pd.DataFrame(scaler_x.transform(X_test_A), columns=X_test_A.columns)
    
    return X_train_A, X_test_A, y_train, y_test, scaler_y

def apply_scaling(X_train_A, X_test_A, y_train, y_test, scaling_method='standard'):
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'robust': RobustScaler(),
        'normalize': Normalizer()  
    }
    
    if scaling_method not in scalers:
        raise ValueError(f"Unsupported scaling method: {scaling_method}")
    
    if scaling_method == 'normalize':
        X_train_A, X_test_A = apply_normalization(X_train_A, X_test_A)
        return X_train_A, X_test_A, y_train, y_test, None  
    
    scaler_y = scalers[scaling_method]
    scaler_y.fit(y_train.values.reshape(-1, 1))
    y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    scaler_x = scalers[scaling_method]
    scaler_x.fit(X_train_A)
    X_train_A = pd.DataFrame(scaler_x.transform(X_train_A), columns=X_train_A.columns)
    X_test_A = pd.DataFrame(scaler_x.transform(X_test_A), columns=X_test_A.columns)
    
    return X_train_A, X_test_A, y_train, y_test, scaler_y

def build_model(hp, input_dim_A, input_dim_B):
    input_A = Input(shape=(input_dim_A,))
    x = input_A

    input_B = Input(shape=(input_dim_B, 1))
    
    # Define the hyperparameters
    num_conv_layers = hp.Int('num_conv_layers', 1, 4, default=1)
    kernel_size = hp.Int('kernel_size', 51, 201, step=10, default=101)
    stride_size = hp.Int('stride_size', 26, 101, step=5, default=51)
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.05, default=0.1)
    use_max_pooling = hp.Boolean('use_max_pooling', default=False)
    num_filters = hp.Int('num_filters', 50, 100, step=10, default=50)

    y = input_B
    for i in range(num_conv_layers):
        y = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=stride_size,
            activation='relu',
            padding='same')(y)
        
        # Ensure the input size is appropriate for max pooling
        if use_max_pooling and y.shape[1] > 1:
            y = MaxPooling1D(pool_size=2)(y)
        
        y = Dropout(dropout_rate)(y)

    y = Flatten()(y)
    y = Dense(4, activation="relu", name='output_B')(y)

    con = concatenate([x, y])  # Corrected concatenate usage

    z = Dense(
        hp.Int('dense', 4, 640, step=32, default=256),
        activation='relu')(con)
    z = Dropout(hp.Float('dropout-2', 0.0, 0.5, step=0.05, default=0.0))(z)

    output = Dense(1, activation="linear")(z)
    model = Model(inputs=[input_A, input_B], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model


def train_and_optimize_model(tuner, X_train_A, X_train_B, y_train, nb_epoch, batch_size):
    outputFilePath = 'Estimator'
    checkpointer = ModelCheckpoint(filepath=outputFilePath, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)

    tuner.search([X_train_A, X_train_B], y_train,
                 epochs=nb_epoch,
                 batch_size=batch_size,
                 shuffle=True,
                 validation_split=0.25,
                 verbose=1,
                 callbacks=[earlystop])

    model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]

    return model, best_hp

def final_training_pass(model, X_train_A, X_train_B, y_train, nb_epoch, batch_size):
    outputFilePath = 'Estimator'
    checkpointer = ModelCheckpoint(filepath=outputFilePath, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)

    history = model.fit([X_train_A, X_train_B], y_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.25,
                        verbose=1,
                        callbacks=[checkpointer, earlystop]).history
    
    return history

def plot_training_history(history):
    plt.figure(figsize=(10, 10))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def evaluate_model(model, X_test_A, X_test_B, y_test):
    evaluation = model.evaluate([X_test_A, X_test_B], y_test)
    preds = model.predict([X_test_A, X_test_B])
    r2 = r2_score(y_test, preds)
    return evaluation, preds, r2


def plot_predictions(y_test, preds):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    lims = [-2.5, 5]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()

def plot_prediction_error(preds, y_test):
    # Ensure preds is a numpy array
    preds = np.array(preds).flatten()
    
    # Convert y_test to a numpy array and flatten it
    y_test = y_test.to_numpy().flatten()
    
    error = preds - y_test

    plt.figure(figsize=(6, 6))
    plt.hist(error, bins=20)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.show()


def evaluate_training_set(model, X_train_A, X_train_B, y_train, scaler_y, f_train):

    
    y_train = np.array(y_train).reshape(-1, 1)

    preds_t = model.predict([X_train_A, X_train_B])
    
    # Ensure preds_t and y_train have the right shape for inverse transform
    preds_t = preds_t.reshape(-1, 1)
    y_train_reshaped = y_train.reshape(-1, 1)
    
    # Inverse transform the predictions and true values
    y_pr_transformed = scaler_y.inverse_transform(preds_t)
    y_tr_transformed = scaler_y.inverse_transform(y_train_reshaped)

    # Calculate r-squared and RMSE
    r_squared_tr = r2_score(y_tr_transformed, y_pr_transformed)
    rmse_tr = sqrt(mean_squared_error(y_tr_transformed, y_pr_transformed))

    # Create DataFrame for predictions and true values
    y_tr_df = pd.DataFrame(y_tr_transformed, columns=['train'])
    y_tr_df['pred'] = y_pr_transformed
    f_train_reset = f_train.reset_index(drop=True)
    y_tr_df['file'] = f_train_reset

    # Save predictions to CSV
    y_tr_df.to_csv('./Output/Data/train_predictions.csv', index=False)

    return r_squared_tr, rmse_tr, y_tr_df



def plot_training_set(y_tr_transformed, y_pr_transformed):
    sns.set_style("white")
    sns.set(style="ticks")
    sns.set_context("poster")

    f, ax = plt.subplots(figsize=(12, 12))
    p = sns.regplot(x=y_tr_transformed, y=y_pr_transformed, ci=None,
                    scatter_kws={"edgecolor": 'b', 'linewidths': 2, "alpha": 0.5, "s": 150},
                    line_kws={"alpha": 0.5, "lw": 4})
    ax.plot([y_tr_transformed.min(), y_tr_transformed.max()], [y_tr_transformed.min(), y_tr_transformed.max()], 'k--', lw=2)

    p.set(xlim=(-1, 24))
    p.set(ylim=(-1, 24))
    sns.despine()
    plt.title('Training Set', fontsize=25)
    plt.xlabel('Traditional Age (years)')
    plt.ylabel('FT-NIR Age (years)')
    plt.savefig('./Output/Figures/TrainingSet.png')
    plt.show()

def plot_test_set(y_test_transformed, y_pred_transformed):
    f, ax = plt.subplots(figsize=(12, 12))
    p = sns.regplot(x=y_test_transformed, y=y_pred_transformed, ci=None,
                    scatter_kws={"edgecolor": 'b', 'linewidths': 2, "alpha": 0.5, "s": 150},
                    line_kws={"alpha": 0.5, "lw": 4})
    ax.plot([y_test_transformed.min(), y_test_transformed.max()], [y_test_transformed.min(), y_test_transformed.max()], 'k--', lw=2)

    p.set(xlim=(-1, 24))
    p.set(ylim=(-1, 24))
    sns.despine()
    plt.title('Test Set', fontsize=25)
    plt.xlabel('Traditional Age (years)')
    plt.ylabel('FT-NIR Age (years)')
    plt.savefig('./Output/Figures/TestSet.png')
    plt.show()

def plot_bland_altman(y_test_transformed, y_pred_transformed):
    pyCompare.blandAltman(y_test_transformed.flatten(), y_pred_transformed.flatten(),
                          limitOfAgreement=1.96, confidenceInterval=95,
                          confidenceIntervalMethod='approximate',
                          detrend=None, percentage=False,
                          title='Bland-Altman Plot\n',
                          savePath='./Output/Figures/BlandAltman.png')



def TrainingModeWithHyperband(filepath, filter_CHOICE, scaling_CHOICE):
    data, X_train, X_test, y_train, y_test, f_train, f_test, s_train, s_test, X_train_A, X_test_A, X_train_B, X_test_B = read_and_clean_data(filepath)
    
    X_train_B, X_test_B = preprocess_spectra(X_train_B, X_test_B, filter_type=filter_CHOICE)
    

    scaling_method = scaling_CHOICE  # 'minmax', 'standard', 'maxabs', 'robust', or 'normalize' 
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y = apply_scaling(X_train_A, X_test_A, y_train, y_test, scaling_method)
    
    input_dim_A = X_train_A.shape[1]
    input_dim_B = X_train_B.shape[1]
    
    def model_builder(hp):
        return build_model(hp, input_dim_A, input_dim_B)
    
    tuner = Hyperband(
        model_builder,
        objective='val_loss',
        max_epochs=1, # !@! 
        directory='Tuners',
        project_name='mmcnn',
        seed=42
    )
    
    print(tuner.search_space_summary())
    
    nb_epoch = 200
    batch_size = 32
    model, best_hp = train_and_optimize_model(tuner, X_train_A, X_train_B, y_train, nb_epoch, batch_size)
    
    nb_epoch = 2000
    batch_size = 32
    history = final_training_pass(model, X_train_A, X_train_B, y_train, nb_epoch, batch_size)
    
    plot_training_history(history)
    
    evaluation, preds, r2 = evaluate_model(model, X_test_A, X_test_B, y_test)
    print(f"Evaluation: {evaluation}, R2: {r2}")
    
    plot_predictions(y_test, preds)
    plot_prediction_error(preds, y_test)
    
    model.summary()

    # plot_model(model, show_shapes=True)
    
    # r_squared_tr, rmse_tr, y_tr_df = evaluate_training_set(model, X_train_A, X_train_B, y_train, scaler_y, f_train)
    # print(f"Training R2: {r_squared_tr}, RMSE: {rmse_tr}")
    
    # plot_training_set(y_tr_df['train'], y_tr_df['pred'])
    
    # y_pred_transformed = scaler_y.inverse_transform(preds.reshape(-1, 1))
    # y_test_transformed = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # r_squared = r2_score(y_test_transformed, y_pred_transformed)
    # rmse = sqrt(mean_squared_error(y_test_transformed, y_pred_transformed))
    # print(f"Test R2: {r_squared}, RMSE: {rmse}")
    
    # y_test_df = pd.DataFrame(y_test_transformed, columns=['test'])
    # y_test_df['pred'] = y_pred_transformed
    # f_test_reset = f_test.reset_index(drop=True)
    # y_test_df['file'] = f_test_reset
    # y_test_df.to_csv('./Output/Data/test_predictions.csv', index=False)
    
    # plot_test_set(y_test_transformed, y_pred_transformed)
    # plot_bland_altman(y_test_transformed, y_pred_transformed)

def build_model_manual(input_dim_A, input_dim_B, num_conv_layers, kernel_size, stride_size, dropout_rate, use_max_pooling, num_filters, dense_units, dropout_rate_2):
    input_A = Input(shape=(input_dim_A,))
    x = input_A

    input_B = Input(shape=(input_dim_B, 1))
    y = input_B
    for i in range(num_conv_layers):
        y = Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=stride_size,
            activation='relu',
            padding='same')(y)
        
        # Ensure the input size is appropriate for max pooling
        if use_max_pooling and y.shape[1] > 1:
            y = MaxPooling1D(pool_size=2)(y)
        
        y = Dropout(dropout_rate)(y)

    y = Flatten()(y)
    y = Dense(4, activation="relu", name='output_B')(y)

    con = concatenate([x, y])  # Corrected concatenate usage

    z = Dense(dense_units, activation='relu')(con)
    z = Dropout(dropout_rate_2)(z)

    output = Dense(1, activation="linear")(z)
    model = Model(inputs=[input_A, input_B], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model


def TrainingModeWithoutHyperband(filepath, num_conv_layers, kernel_size, stride_size, dropout_rate, use_max_pooling, num_filters, dense_units, dropout_rate_2, filter_CHOICE, scaling_CHOICE):
    data, X_train, X_test, y_train, y_test, f_train, f_test, s_train, s_test, X_train_A, X_test_A, X_train_B, X_test_B = read_and_clean_data(filepath)
    
    X_train_B, X_test_B = preprocess_spectra(X_train_B, X_test_B, filter_type=filter_CHOICE)
    
    scaling_method = scaling_CHOICE  # 'minmax', 'standard', 'maxabs', 'robust', or 'normalize'
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y = apply_scaling(X_train_A, X_test_A, y_train, y_test, scaling_method)
    
    input_dim_A = X_train_A.shape[1]
    input_dim_B = X_train_B.shape[1]

    model = build_model_manual(
        input_dim_A,
        input_dim_B,
        num_conv_layers,
        kernel_size,
        stride_size,
        dropout_rate,
        use_max_pooling,
        num_filters,
        dense_units,
        dropout_rate_2
    )
    
    nb_epoch = 200
    batch_size = 32
    history = final_training_pass(model, X_train_A, X_train_B, y_train, nb_epoch, batch_size)
    
    plot_training_history(history)
    
    evaluation, preds, r2 = evaluate_model(model, X_test_A, X_test_B, y_test)
    print(f"Evaluation: {evaluation}, R2: {r2}")
    
    plot_predictions(y_test, preds)
    plot_prediction_error(preds, y_test)
    
    model.summary()



def InferenceMode(model_path, image_path):
    # Load the model from the given path
    model = load_model(model_path)
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))  # Adjust target size according to your model's input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image array

    # Run inference
    predictions = model.predict(img_array)

    return predictions


def main():

    TrainingModeWithHyperband(
        filepath='./Data/AGP_MMCNN_BSsurvey_pollock2014to2018.csv',
        filter_CHOICE='savgol',
        scaling_CHOICE='minmax'

        # can set hyperparameter ranges?
        # make it optional
        # only allow certain parameters and constrain the search space...
    
    )

    TrainingModeWithoutHyperband(
        filepath='./Data/AGP_MMCNN_BSsurvey_pollock2014to2018.csv',
        num_conv_layers=2,
        kernel_size=101,
        stride_size=51,
        dropout_rate=0.1,
        use_max_pooling=False,
        num_filters=50,
        dense_units=256,
        dropout_rate_2=0.1,
        filter_CHOICE='savgol',
        scaling_CHOICE='minmax'
    )

    inference_output = InferenceMode(model_path = '', image_path = '')

    # put acceptable ranges for size variables and enums for enum variables...
    # just put in comments

    # have functions read from in memory object rather than filepath



   

if __name__ == "__main__":
    main()
