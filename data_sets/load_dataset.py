from datetime import datetime
import numpy as np
import pandas as pd
import os


def latex_name(name: str) -> str:
    """
    Converts a dataset string into a latex command.
    :param name:
        name of the dataset
    :return:
        a latex command for the dataset
    """
    if name.startswith('tamilnadu_electricity'):
        name = 'tamilnadu'
    elif name.startswith('pm25'):
        name = 'pm'
    return '\\' + name.upper() + '{}'


def load_dataset(name: str) -> np.ndarray:
    """
    Loads a dataset and returns a standardized numpy array.
    :param name:
        name of the dataset
    :return:
        standardized dataset as (N, D) array, where N is the size
    """

    # the zeros_ dataset can not be standardized
    if name.startswith('zeros_'):
        _, N = name.split('zeros_')
        return np.zeros((int(N), 1))

    return standardize(load_raw_dataset(name))


def load_raw_dataset(name: str) -> np.ndarray:
    """
    Loads a dataset without standardizing it.
    :param name:
        name of the dataset
    :return:
        (N, D) array, where N is the size
    """
    data_dir = os.path.dirname(__file__)
    if name == 'pm25':
        def parser(y, m, d, h):
            def check(x):
                if len(x) == 1:
                    return '0'+x
                return x
            return [datetime.strptime(str(yx)+check(mx)+check(dx)+check(hx), "%Y%m%d%H") for (yx, mx, dx, hx) in zip(y, m, d, h)]

        X = pd.read_csv(os.path.join(data_dir, os.path.join('beijing_pm25', 'PRSA_data_2010.1.1-2014.12.31.csv')),
                          sep=',', parse_dates={0: [1, 2, 3, 4]}, usecols=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
                          date_parser=parser, header=None, skiprows=1)
        X[0] = X[0].astype(int) / 10**11  # convert time to continuous variable
        X[0] = X[0] - X[0][0]
        X = pd.get_dummies(X)
        X = X.values
    elif name == 'bank':
        data_frame = pd.read_csv(os.path.join(data_dir, os.path.join('bank_marketing', 'bank-full.csv')),
                                 sep=';', header=None, skiprows=1, usecols=range(0, 16))
        data_frame = pd.get_dummies(data_frame)
        X = data_frame.values
    elif name == 'protein':
        X = pd.read_csv(os.path.join(data_dir, os.path.join('protein_tertiary', 'CASP.csv')), sep=',',
                                   header=None, skiprows=1)
        X = X.values[:, 1:]
    elif name == 'tamilnadu_electricity':
        from scipy.io import arff
        data_frame = pd.DataFrame(arff.loadarff(os.path.join(data_dir, os.path.join('tamilnadu_electricity', 'eb.arff')))[0])
        del data_frame['Sector']  # all values are the same
        data_frame = pd.get_dummies(data_frame)
        X = data_frame.values
    elif name == 'metro':
        csv = pd.read_csv(os.path.join(data_dir, os.path.join('metro', 'Metro_Interstate_Traffic_Volume.csv')),
                          sep=',', skiprows=1, header=None, parse_dates=[7], usecols=range(0, 8))
        csv[7] = csv[7].astype(int) / 10**11  # convert time to continuous variable
        csv[7] = csv[7] - csv[7][0]

        csv = pd.get_dummies(csv)
        X = csv.values
    elif name == 'pumadyn':
        from scipy.io import loadmat
        puma = loadmat(os.path.join(data_dir, os.path.join('pumadyn', 'pumadyn32nm.mat')))
        X = np.vstack([puma['X_tr'], puma['X_tst']])
    else:
        raise ValueError('unknown dataset:', name)
    return X


def standardize(x_train: np.ndarray) -> np.ndarray:
    """
    standardizes a given (N, D) array along the first dimension
    :param x_train:
        the (N, D) array
    :return:
        the standardized (N, D) array
    """
    stdTrainX = np.std(x_train, axis=0)

    x_train = x_train[:, stdTrainX != 0.]  # remove columns with 0 variance
    stdTrainX = stdTrainX[stdTrainX != 0.]

    mean_x = np.mean(x_train, axis=0)
    x_train = x_train - mean_x
    x_train = x_train / stdTrainX

    if np.isnan(x_train).any() or np.isinf(x_train).any():
        raise ValueError("standardized dataset contains NaNs!")
    return x_train
