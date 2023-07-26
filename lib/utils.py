import torch
from torch import nn
from math import log10
import numpy as np
from os.path import isfile

def load_checkpoint(filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        print("=> loaded checkpoint '{}'"
              .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return checkpoint
