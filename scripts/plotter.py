import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
import sys
 
if __name__ == '__main__':


    with open(sys.argv[1]+'.pkl', 'rb') as handle:
        accs, losses, parameters, probs_layer_1, depths, archs = pkl.load(handle)

    print(parameters)
    # Smoothes out the plot a bit
    N = 100
    cumsum, moving_aves = [0], []
    for i, x in enumerate(accs, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)

    N = 30
    cumsum, moving_aves_std = [0], []
    for i, x in enumerate(accs, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves_std.append(moving_ave)
    moving_aves = np.array(moving_aves)
    moving_aves_std = np.array(moving_aves_std)
    #accs = np.convolve(accs, np.ones((5,))/5, mode='valid')
    losses = np.convolve(losses, np.ones((5,))/5, mode='valid')
    print("acc:", np.sum(accs[-100:])/len(accs[-100:]))
    # Generate the figure
    sns.set(style='darkgrid')
    plt.figure()
    plt.plot(range(len(moving_aves_std)),moving_aves_std,alpha=0.3)
    plt.plot(range(len(moving_aves)), moving_aves,alpha=1,color='orange')
    plt.title("The accuracy for the selected architectures", fontsize=18)
    plt.xlabel('Rollout', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Rollout')
    plt.ylabel('losses')

    plt.figure()
    plt.plot(range(len(depths)), depths)
    plt.xlabel('Rollout')
    plt.ylabel('Depth (including acti)')


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
    plt.title("The probability of taking the given actions at state 0", fontsize=18)
    plt.xlabel('Rollout', fontsize=14)
    plt.ylabel('Chance', fontsize=14)
    plt.show()
