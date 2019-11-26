import pickle as pkl
import numpy as np
import torch
from trainer import *
from os import sys
import os
from datetime import datetime


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        rollouts = int(sys.argv[1])
        dataset = str(sys.argv[2])
        lr = float(sys.argv[3])
    else:
        raise Exception("YOU ARE WRONG!")
    # A number of output files and folders contain a timestamp as part of their name.
    timestamp = datetime.now().strftime("%H%M%S-%d%m%Y")
    os.makedirs("../runs/"+timestamp)
    accuracy_hist, loss_hist = trainer(rollouts,dataset,lr)

    with open("../runs/" + timestamp + '/rewards_losses.pkl', 'wb') as handle:
        pkl.dump((accuracy_hist, loss_hist), handle, protocol=pkl.HIGHEST_PROTOCOL)
