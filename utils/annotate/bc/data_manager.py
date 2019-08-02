import numpy as np
import os
import pickle
path = os.path.abspath(__file__)
datapath = os.path.dirname(path)
datapath = ''
def load_data(file1='wiki.npz', file2='wiki_label.npz', datapath=datapath):
    print('Start loading...')
    data = np.load(os.path.join(datapath,file1))
    X_train_qu = data['X_train_qu']
    X_train_col = data['X_train_col']
    X_test_qu = data['X_test_qu']
    X_test_col = data['X_test_col']
    X_dev_qu = data['X_dev_qu']
    X_dev_col = data['X_dev_col']
    print('Finish loading...')

    data = np.load(os.path.join(datapath,file2))
    
    y_train = data['y_train'].reshape(-1, 1)
    y_test = data['y_test'].reshape(-1, 1)
    y_dev = data['y_dev'].reshape(-1, 1)


    return X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev


if __name__ == '__main__':
    X_train_qu, X_train_col, y_train, X_test_qu, X_test_col, y_test, X_dev_qu, X_dev_col, y_dev = load_data()
    