import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import pyCompare
import matplotlib
import keras_tuner as kt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, Normalizer, RobustScaler, MaxAbsScaler
from yellowbrick.regressor import PredictionError, ResidualsPlot
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from keras_tuner.tuners import BayesianOptimization, Hyperband
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU, Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter
import pywt
from sklearn.decomposition import PCA
import sys

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

sns.set(rc={'figure.figsize': (10, 6)})
sns.set(style="whitegrid", font_scale=2)

# Read and Clean Data
def read_and_clean_data(filepath, drop_outliers=True):
    data = pd.read_csv(filepath)
    
    if drop_outliers:
        data = data[data['sample'] != 'outlier']
    
    data.rename(columns={"final_age": "Age"}, inplace=True)
    
    if data.isnull().values.any():
        raise ValueError("Data contains NaN values")

    print(data.head())
    print(list(data.columns[0:10])) # ['file_name', 'sample', 'Age', 'latitude', 'length', 'gear_depth', 'gear_temp', 'wn11476.85064', 'wn11468.60577', 'wn11460.36091']
    return data

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
def preprocess_spectra(data, filter_type='savgol'):
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
    
    data.loc[data['sample'] == 'training', data.columns[4:]] = filter_func(data.loc[data['sample'] == 'training', data.columns[4:]].values)
    data.loc[data['sample'] == 'test', data.columns[4:]] = filter_func(data.loc[data['sample'] == 'test', data.columns[4:]].values)
    
    return data

def apply_normalization(data, columns):
    normalizer = Normalizer()
    data[columns] = normalizer.fit_transform(data[columns])
    return data

def apply_robust_scaling(data, y_col, feature_columns):
    scaler_y = RobustScaler()
    data[y_col] = scaler_y.fit_transform(data[[y_col]])
    data[feature_columns] = data[feature_columns].apply(lambda col: RobustScaler().fit_transform(col.values.reshape(-1, 1)))
    return data, scaler_y

def apply_minmax_scaling(data, y_col, feature_columns):
    scaler_y = MinMaxScaler()
    data[y_col] = scaler_y.fit_transform(data[[y_col]])
    data[feature_columns] = data[feature_columns].apply(lambda col: MinMaxScaler().fit_transform(col.values.reshape(-1, 1)))
    return data, scaler_y

def apply_maxabs_scaling(data, y_col, feature_columns):
    scaler_y = MaxAbsScaler()
    data[y_col] = scaler_y.fit_transform(data[[y_col]])
    data[feature_columns] = data[feature_columns].apply(lambda col: MaxAbsScaler().fit_transform(col.values.reshape(-1, 1)))
    return data, scaler_y

def apply_scaling(data, scaling_method='standard', y_col='Age'):
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'maxabs': MaxAbsScaler(),
        'robust': RobustScaler(),
        'normalize': Normalizer()  
    }
    
    if scaling_method not in scalers:
        raise ValueError(f"Unsupported scaling method: {scaling_method}")
    
    feature_columns = data.columns.difference(['sample', 'file_name', y_col])

    if scaling_method == 'normalize':
        data = apply_normalization(data, feature_columns)
        return data, None  
    
    scaler_y = scalers[scaling_method]
    data[y_col] = scaler_y.fit_transform(data[[y_col]])
    scaler_x = scalers[scaling_method]
    data[feature_columns] = scaler_x.fit_transform(data[feature_columns])
    
    return data, scaler_y

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

    con = concatenate([x, y])

    z = Dense(
        hp.Int('dense', 4, 640, step=32, default=256),
        activation='relu')(con)
    z = Dropout(hp.Float('dropout-2', 0.0, 0.5, step=0.05, default=0.0))(z)

    output = Dense(1, activation="linear")(z)
    model = Model(inputs=[input_A, input_B], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model

def train_and_optimize_model(tuner, data, nb_epoch, batch_size):
    outputFilePath = 'Estimator'
    checkpointer = ModelCheckpoint(filepath=outputFilePath, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)

    X_train_biological_data = data.loc[data['sample'] == 'training', data.columns[4:8]]
    X_train_wavenumbers = data.loc[data['sample'] == 'training', data.columns[8:]]
    y_train = data.loc[data['sample'] == 'training', 'Age']

    tuner.search([X_train_biological_data, X_train_wavenumbers], y_train,
                 epochs=nb_epoch,
                 batch_size=batch_size,
                 shuffle=True,
                 validation_split=0.25,
                 verbose=1,
                 callbacks=[earlystop])

    model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters()[0]

    return model, best_hp

def final_training_pass(model, data, nb_epoch, batch_size):
    outputFilePath = 'Estimator'
    checkpointer = ModelCheckpoint(filepath=outputFilePath, verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)

    X_train_biological_data = data.loc[data['sample'] == 'training', data.columns[4:8]]
    X_train_wavenumbers = data.loc[data['sample'] == 'training', data.columns[8:]]
    y_train = data.loc[data['sample'] == 'training', 'Age']

    history = model.fit([X_train_biological_data, X_train_wavenumbers], y_train,
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

def evaluate_model(model, data):
    X_test_biological_data = data.loc[data['sample'] == 'test', data.columns[4:8]]
    X_test_wavenumbers = data.loc[data['sample'] == 'test', data.columns[8:]]
    y_test = data.loc[data['sample'] == 'test', 'Age']

    evaluation = model.evaluate([X_test_biological_data, X_test_wavenumbers], y_test)
    preds = model.predict([X_test_biological_data, X_test_wavenumbers])
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
    preds = np.array(preds).flatten()
    y_test = y_test.to_numpy().flatten()
    error = preds - y_test

    plt.figure(figsize=(6, 6))
    plt.hist(error, bins=20)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.show()

def evaluate_training_set(model, data, scaler_y):
    X_train_biological_data = data.loc[data['sample'] == 'training', data.columns[4:8]]
    X_train_wavenumbers = data.loc[data['sample'] == 'training', data.columns[8:]]
    y_train = data.loc[data['sample'] == 'training', 'Age']
    f_train = data.loc[data['sample'] == 'training', 'file_name']

    y_train = np.array(y_train).reshape(-1, 1)
    preds_t = model.predict([X_train_biological_data, X_train_wavenumbers])
    
    preds_t = preds_t.reshape(-1, 1)
    y_train_reshaped = y_train.reshape(-1, 1)
    
    y_pr_transformed = scaler_y.inverse_transform(preds_t)
    y_tr_transformed = scaler_y.inverse_transform(y_train_reshaped)

    r_squared_tr = r2_score(y_tr_transformed, y_pr_transformed)
    rmse_tr = sqrt(mean_squared_error(y_tr_transformed, y_pr_transformed))

    y_tr_df = pd.DataFrame(y_tr_transformed, columns=['train'])
    y_tr_df['pred'] = y_pr_transformed
    y_tr_df['file'] = f_train.reset_index(drop=True)

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
        
        if use_max_pooling and y.shape[1] > 1:
            y = MaxPooling1D(pool_size=2)(y)
        
        y = Dropout(dropout_rate)(y)

    y = Flatten()(y)
    y = Dense(4, activation="relu", name='output_B')(y)

    con = concatenate([x, y])

    z = Dense(dense_units, activation='relu')(con)
    z = Dropout(dropout_rate_2)(z)

    output = Dense(1, activation="linear")(z)
    model = Model(inputs=[input_A, input_B], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model

def InferenceMode(model_path, image_path):
    model = load_model(model_path)
    
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)

    return predictions

def TrainingModeWithHyperband(filepath, filter_CHOICE, scaling_CHOICE):
    data = read_and_clean_data(filepath)
    
    data = preprocess_spectra(data, filter_type=filter_CHOICE)
    
    y_col = 'Age'
    scaling_method = scaling_CHOICE  # 'minmax', 'standard', 'maxabs', 'robust', or 'normalize'
    data, scaler_y = apply_scaling(data, scaling_method, y_col)

    input_dim_A = data.columns[4:8].shape[0]
    input_dim_B = data.columns[8:].shape[0]
    
    def model_builder(hp):
        return build_model(hp, input_dim_A, input_dim_B)
    
    tuner = Hyperband(
        model_builder,
        objective='val_loss',
        max_epochs=1,
        directory='Tuners',
        project_name='mmcnn',
        seed=42
    )
    
    print(tuner.search_space_summary())
    
    nb_epoch = 1  # !@!
    batch_size = 32
    model, best_hp = train_and_optimize_model(tuner, data, nb_epoch, batch_size)
    
    nb_epoch = 1  # !@!
    batch_size = 32
    history = final_training_pass(model, data, nb_epoch, batch_size)
    
    evaluation, preds, r2 = evaluate_model(model, data)
    print(f"Evaluation: {evaluation}, R2: {r2}")
    
    model.summary()
    
    training_outputs = {
        'trained_model': model,
        'training_history': history,
        'evaluation': evaluation,
        'predictions': preds,
        'r2_score': r2
    }
    
    top_5_models = tuner.get_best_models(num_models=5)
    
    additional_outputs = {
        'top5models': top_5_models
    }
    
    return training_outputs, additional_outputs


def TrainingModeWithoutHyperband(filepath, filter_CHOICE, scaling_CHOICE, hyperband_parameters):
    if len(hyperband_parameters) != 8:
        raise ValueError("hyperband_parameters must be a list of 8 values.")
    
    num_conv_layers, kernel_size, stride_size, dropout_rate, use_max_pooling, num_filters, dense_units, dropout_rate_2 = hyperband_parameters
    
    if not all(isinstance(param, (int, float, bool)) for param in hyperband_parameters):
        raise ValueError("All hyperband parameters must be either int, float, or bool.")
    
    data = read_and_clean_data(filepath)
    
    data = preprocess_spectra(data, filter_type=filter_CHOICE)
    
    y_col = 'Age'
    scaling_method = scaling_CHOICE  # 'minmax', 'standard', 'maxabs', 'robust', or 'normalize'
    data, scaler_y = apply_scaling(data, scaling_method, y_col)

    input_dim_A = data.columns[4:8].shape[0]
    input_dim_B = data.columns[8:].shape[0]

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
    
    nb_epoch = 1  # !@!
    batch_size = 32
    history = final_training_pass(model, data, nb_epoch, batch_size)
    
    evaluation, preds, r2 = evaluate_model(model, data)
    print(f"Evaluation: {evaluation}, R2: {r2}")
    
    model.summary()
    
    training_outputs = {
        'trained_model': model,
        'training_history': history,
        'evaluation': evaluation,
        'predictions': preds,
        'r2_score': r2
    }
    
    additional_outputs = {
        
    }
    
    return training_outputs, additional_outputs


def main():
    training_outputs_hyperband, additional_outputs_hyperband = TrainingModeWithHyperband(
        filepath='./Data/AGP_MMCNN_BSsurvey_pollock2014to2018.csv',
        filter_CHOICE='savgol',
        scaling_CHOICE='minmax'
    )

    training_outputs_manual, additional_outputs_manual = TrainingModeWithoutHyperband(
        filepath='./Data/AGP_MMCNN_BSsurvey_pollock2014to2018.csv',
        filter_CHOICE='savgol',
        scaling_CHOICE='minmax',
        hyperband_parameters=[2, 101, 51, 0.1, False, 50, 256, 0.1]
    )

    # Use the outputs as needed
    print(training_outputs_hyperband)
    print(additional_outputs_hyperband)
    print(training_outputs_manual)
    print(additional_outputs_manual)

    # inference_output = InferenceMode(model_path='', image_path='')


if __name__ == "__main__":
    main()