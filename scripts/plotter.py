import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns


if __name__ == '__main__':

    with open('rewards_losses.pkl', 'rb') as handle:
        accs, losses = pkl.load(handle)

    # Smoothes out the plot a bit
    #accs = np.convolve(accs, np.ones((5,))/5, mode='valid')
    #losses = np.convolve(losses, np.ones((5,))/5, mode='valid')

    # Generate the figure
    sns.set(style='darkgrid')
    plt.figure()
    plt.plot(range(len(accs)), accs)
    plt.xlabel('Rollout')
    plt.ylabel('Accuracy')
    
    sns.set(style='darkgrid')
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Rollout')
    plt.ylabel('losses')
    plt.show()
    plt.show()