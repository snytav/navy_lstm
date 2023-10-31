import seaborn as sns
import matplotlib.pyplot as plt


def plot_close_column(data):
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 9))
    plt.plot(data[['Close']])
    plt.xticks(range(0, data.shape[0], 500), data['Date'].loc[::500], rotation=45)
    plt.title("Amazon Stock Price", fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price (USD)', fontsize=18)
    plt.show()


def prediction_convergence_plot(original,predict,hist):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
    ax.set_title('Stock price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (USD)", size = 14)
    ax.set_xticklabels('', size=10)


    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epoch", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Training Loss", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)