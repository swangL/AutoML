import pickle as pkl
import numpy as np
import torch
from trainer import *
from os import sys
import os
from datetime import datetime


if __name__ == "__main__":
    parameters = {}
    if len(sys.argv) >= 5:
        parameters["rollouts"] = int(sys.argv[1])
        parameters["dataset"] = str(sys.argv[2])
        parameters["lr"] = float(sys.argv[3])
        parameters["controltype"] = str(sys.argv[4])
    else:
        raise Exception("YOU ARE WRONG!")
    # A number of output files and folders contain a timestamp as part of their name.
    timestamp = datetime.now().strftime("%H%M%S-%d%m%Y")
    accuracy_hist, loss_hist, probs_layer_1, depth, sample_networks = trainer(parameters["rollouts"],parameters["dataset"], parameters["lr"], parameters["controltype"])
    with open("../runs/conv2_" + timestamp + '.pkl', 'wb') as handle:
        pkl.dump((accuracy_hist, loss_hist, parameters, probs_layer_1, depth, sample_networks), handle, protocol=pkl.HIGHEST_PROTOCOL)
