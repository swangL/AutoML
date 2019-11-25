from torch.autograd import Variable
import torch

def get_variable(x):
    if torch.cuda.is_available():
        return Variable(x).cuda()
    else:
        return Variable(x)
