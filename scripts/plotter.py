import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
import sys

if __name__ == '__main__':

    with open(sys.argv[1]+'.pkl', 'rb') as handle:
        accs, losses, parameters, probs_layer_1 = pkl.load(handle)

    print(parameters)
    # Smoothes out the plot a bit
    accs = np.convolve(accs, np.ones((5,))/5, mode='valid')
    losses = np.convolve(losses, np.ones((5,))/5, mode='valid')

    # Generate the figure
    sns.set(style='darkgrid')
    plt.figure()
    plt.plot(range(len(accs)), accs)
    plt.xlabel('Rollout')
    plt.ylabel('Accuracy')
    
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Rollout')
    plt.ylabel('losses')

    sizes_dict={
        0:"term",
        1:2,
        2:4,
        3:8,
        4:16,
        5:32,
        6:64,
    }
    data = np.array(probs_layer_1).T.tolist()
    plt.figure()
    plt.stackplot(range(1,len(data[0])+1), *data)
    plt.legend([sizes_dict[i] for i in range(len(sizes_dict))])
    plt.xlabel('Episode')
    plt.ylabel('Chance')
    plt.show()