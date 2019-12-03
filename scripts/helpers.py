from torch.autograd import Variable
import torch
 
def get_variable(x, **kwargs):
    if torch.cuda.is_available():
        return Variable(x, **kwargs).cuda()
    else:
        return Variable(x, **kwargs)
