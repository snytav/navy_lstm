

import numpy as np
import pandas as pd
data = pd.read_csv('AMZN_2006-01-01_to_2018-01-01.csv')
data.head()



import matplotlib.pyplot as plt
import seaborn as sns

from neural_plot import plot_close_column

plot_close_column(data)

"""## Normalize data"""

price = data[['Close']]
price.info()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))

from split_func import split_data

lookback = 20 # choose sequence length
x_train, y_train_lstm, x_test, y_test_lstm = split_data(price, lookback)


import torch
import torch.nn as nn


input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100


def predict(data,hidden_dim,num_layers,num_epochs):

    from LSTM_net import LSTM

    from train_func import train
    model,y_train_pred,hist = train(num_epochs,x_train,y_train_lstm,input_dim, hidden_dim,output_dim, num_layers)

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

    import seaborn as sns
    sns.set_style("darkgrid")

    from neural_plot import prediction_convergence_plot

    prediction_convergence_plot(original,predict,hist)

    from errors import predictions_and_errors
    testScore,mae,mape  = predictions_and_errors(model,x_test,y_train_pred,y_train_lstm,y_test_lstm,scaler)
    print('Test Score: %.2f RMSE %e MAE %e ' % (testScore,mae,mape))
    return mape


if __name__ == '__main__':
    mape = predict(data, hidden_dim, num_layers, num_epochs)