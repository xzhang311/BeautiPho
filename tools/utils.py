import torch
import gc         # garbage collect library
import torch

def del_model(model):
    del model

def release_cache():
    gc.collect()
    torch.cuda.empty_cache() 