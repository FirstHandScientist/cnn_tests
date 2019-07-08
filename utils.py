# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.io import loadmat


def prepare_cifar10():
    cifar10 = loadmat("./mat_files/CIFAR-10.mat")
    X_train =  cifar10["train_x"].astype(np.float32)
    X_test =  cifar10["test_x"].astype(np.float32)
    T_train =  cifar10["train_y"].astype(np.float32)
    T_test=  cifar10["test_y"].astype(np.float32)
    return X_train, X_test, T_train, T_test

def prepare_cifar100():
    X_train =  loadmat("./mat_files/CIFAR-100.mat")["train_x"].astype(np.float32)
    X_test =  loadmat("./mat_files/CIFAR-100.mat")["test_x"].astype(np.float32)
    T_train =  loadmat("./mat_files/CIFAR-100.mat")["train_y_fine"].astype(np.float32)
    T_test=  loadmat("./mat_files/CIFAR-100.mat")["test_y_fine"].astype(np.float32)
    return X_train, X_test, T_train, T_test

def prepare_satimage():
    train_X = loadmat("./mat_files/Satimage.mat")["train_x"].astype(np.float32)
    train_T = loadmat("./mat_files/Satimage.mat")["train_y"].astype(np.float32)
    test_X = loadmat("./mat_files/Satimage.mat")["test_x"].astype(np.float32)
    test_T= loadmat("./mat_files/Satimage.mat")["test_y"].astype(np.float32)
    return train_X, test_X, train_T, test_T

def prepare_mnist():
    X_train =  loadmat("./mat_files/MNIST.mat")["train_x"].astype(np.float32)
    X_test =  loadmat("./mat_files/MNIST.mat")["test_x"].astype(np.float32)
    T_train =  loadmat("./mat_files/MNIST.mat")["train_y"].astype(np.float32)
    T_test=  loadmat("./mat_files/MNIST.mat")["test_y"].astype(np.float32)
    return X_train, X_test, T_train, T_test

def prepare_vowel():
    X = loadmat("./mat_files/Vowel.mat")["featureMat"]
    T = loadmat("./mat_files/Vowel.mat")["labelMat"]
    X_train,X_test = X[:, :528].astype(np.float32), X[:, 528:].astype(np.float32)
    T_train, T_test = T[:, :528].astype(np.float32), T[:, 528:].astype(np.float32)
    return X_train, X_test, T_train, T_test

def prepare_norb():
    train_X = loadmat("./mat_files/NORB.mat")["train_x"].astype(np.float32)
    train_T = loadmat("./mat_files/NORB.mat")["train_y"].astype(np.float32)
    test_X = loadmat("./mat_files/NORB.mat")["test_x"].astype(np.float32)
    test_T= loadmat("./mat_files/NORB.mat")["test_y"].astype(np.float32)
    return train_X, test_X, train_T, test_T

def prepare_shuttle():
    train_X = loadmat("./mat_files/Shuttle.mat")["train_x"].astype(np.float32)
    train_T = loadmat("./mat_files/Shuttle.mat")["train_y"].astype(np.float32)
    test_X = loadmat("./mat_files/Shuttle.mat")["test_x"].astype(np.float32)
    test_T= loadmat("./mat_files/Shuttle.mat")["test_y"].astype(np.float32)
    return train_X, test_X, train_T, test_T

def prepare_yale():
    train_num = 1600
    test_num = 800
    X = loadmat("./mat_files/ExtendedYaleB.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/ExtendedYaleB.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    train_X, test_X = X[:, random_train_lists], X[:, random_test_lists]
    train_T, test_T = T[:, random_train_lists], T[:, random_test_lists]
    return train_X, test_X, train_T, test_T

def prepare_scene():
    train_num = 3000
    test_num = 1400
    X = loadmat("./mat_files/Scene15.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/Scene15.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    train_X, test_X = X[:, random_train_lists], X[:, random_test_lists]
    train_T, test_T = T[:, random_train_lists], T[:, random_test_lists]
    return train_X, test_X, train_T, test_T

def prepare_caltech():
    train_num = 6000
    test_num = 3000
    X = loadmat("./mat_files/Caltech101.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/Caltech101.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    train_X, test_X = X[:, random_train_lists], X[:, random_test_lists]
    train_T, test_T = T[:, random_train_lists], T[:, random_test_lists]
    return train_X, test_X, train_T, test_T
   
def prepare_letter():
    train_num = 13333
    test_num = 6667
    X = loadmat("./mat_files/Letter.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/Letter.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    train_X, test_X = X[:, random_train_lists], X[:, random_test_lists]
    train_T, test_T = T[:, random_train_lists], T[:, random_test_lists]
    return train_X, test_X, train_T, test_T

def prepare_ar():
    train_num = 1800
    test_num = 800
    X = loadmat("./mat_files/AR.mat")["featureMat"].astype(np.float32)
    T = loadmat("./mat_files/AR.mat")["labelMat"].astype(np.float32)
    random_lists = np.random.choice(range(X.shape[1]), train_num + test_num, replace=False)
    random_train_lists = random_lists[:train_num]
    random_test_lists = random_lists[train_num:]
    train_X, test_X = X[:, random_train_lists], X[:, random_test_lists]
    train_T, test_T = T[:, random_train_lists], T[:, random_test_lists]
    return train_X, test_X, train_T, test_T
