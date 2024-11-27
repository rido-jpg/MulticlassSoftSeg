# %%
import sys
import os
from pathlib import Path

file = Path(__file__).resolve()

sys.path.append(str(file.parents[0]))
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

# %%
import torch
import numpy as np
#from utils.brats_tools import postprocessing
# %%
def postprocessing(soft_scores, dec: int):
    '''
    kills all values smaller than one and rounds to a certain number of decimals
    should work for tensors as well as numpy arrays

    '''
    if isinstance(soft_scores, np.ndarray):
        soft_scores = np.maximum(0,soft_scores)
    elif isinstance(soft_scores, torch.Tensor):
        soft_scores = torch.nn.functional.relu(soft_scores)
    soft_scores = soft_scores.round(decimals = dec)
    return soft_scores
# %%
b, c, h, w, d = 1, 2, 2, 2, 2
#tensor = torch.rand((b,c,h,w,d))
tensor = torch.FloatTensor((b,c,h,w,d)).uniform_(-1, 1)
np_array = tensor.numpy()
print(tensor)
# %%
print(postprocessing(tensor, 2))
# %%
print(postprocessing(np_array, 2))
print(np_array.dtype)

# %%
