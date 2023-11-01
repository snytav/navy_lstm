import numpy as np
import torch
from LSTM_net import LSTM

def train(num_epochs,x_train,y_train_lstm,input_dim, hidden_dim,output_dim, num_layers):
    import time

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss(reduction='mean')

    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []

    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))

    return model,y_train_pred,hist[-1]