import torch
import numpy as np

def objective_function(p1, p2, model):
    if model == 'gan':
        G_loss = torch.mean(torch.log(p1) + torch.log(1. - p2))
        D_loss = -G_loss 
    elif model == 'nsgan':
        # The cap is to prevent nan values and is also done internally in pytorch cross entropy.
        G_loss = -torch.mean(torch.log(torch.clamp(p2, min=(10**(-4)))))
        D_loss = -torch.mean(torch.log(torch.clamp(p1, min=(10**(-4)))) + torch.log(1. - torch.clamp(p2, max=(1-10**(-4)))))        
    return G_loss, D_loss
