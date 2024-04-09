import torch
import random
import numpy as np
import tensorflow as tf

def prepare_seed(seed:int=0):
    random.seed(seed) 
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)