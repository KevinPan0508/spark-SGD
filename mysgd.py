import numpy as np
from sklearn.metrics import mean_squared_error
from statistics import mean
import math
The_data = np.loadtxt('./data_banknote_authentication.txt',delimiter=',')

def Logistic(s):
    return np.exp(s)/(1+np.exp(s))

def CE_error(y1,y2):
    error = 0
    for i in range(len(y1)):
        error += math.log(1+math.exp(-y1[i]*y2[i]))

    return error/len(y1)

def Data_Setting(data_bin,x_0):
    data = np.zeros([data_bin.shape[0], data_bin.shape[1] + 1], np.float)
    for i in range(data.shape[0]):
        data[i, 0] = x_0
    for i in range(data_bin.shape[0]):
        for j in range(data_bin.shape[1]):
            data[i, j + 1] = data_bin[i, j]
    examples = data
    x = np.zeros([data.shape[0], data.shape[1] - 1], np.float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1] - 1):
            x[i, j] = data[i, j]
    y = data_bin[:, -1]

    return x,y,examples

def SGD(Eta,data,x_ori,y_ori,W_lin):
    W_t = np.zeros(W_lin.shape, np.float)
    for i in range(10000000):
        #randomly picks an example
        rand_arr = np.arange(data.shape[0])
        np.random.shuffle(rand_arr)
        data_bin = data[rand_arr[0:1]]
        x = np.mat(data_bin[0, 0:-1]).transpose()
        y = np.mat(data_bin[0, -1])
        #
        s = np.dot(np.mat(W_t).transpose(),x)
        W_t = W_t + Eta * Logistic(float(-y * s)) * float(y)*x  #updating  rules

    y_test = np.dot(x_ori,W_t)
    err_ce = CE_error(y_test,y_ori)

    return err_ce


x,y,examples  = Data_Setting(The_data,1)
x_pinv = np.linalg.pinv(x)
W_lin = np.mat(x_pinv)*np.mat(y).transpose()

print(SGD(0.001,examples,x,y,W_lin))