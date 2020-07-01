import torch
from torch import autograd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sys
import os
import datetime
import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
from ComputationalTools import JacobianVectorProduct, SchurComplement
from EigenvalueTools import calc_game_eigs
from LoggingTools import log, plot_log, log_eigs, store_eigs
from GameGradients import build_game_gradient, build_game_jacobian
from LeaderUpdateTools import compute_leader_grad, compute_stackelberg_grad, adam_grad, leader_step
from ObjectiveTools import objective_function
from Utils import calc_gradient_norm, network_param_index, mnist_show
from DCGAN import DCGAN_Generator, DCGAN_Discriminator, DCGAN_weights_init
np.set_printoptions(precision=2)    

seed = int(sys.argv[7])
torch.manual_seed(seed)    
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M")

show = False
sim_name = sys.argv[1]
results_dir = os.path.join(os.getcwd(), 'MNIST_Results')
save_dir = os.path.join(results_dir, '_'.join([sim_name, now]))
fig_dir = os.path.join(save_dir, 'Figs')
info_dir = os.path.join(save_dir, 'Info')
checkpoint_dir = os.path.join(save_dir, 'Checkpoint')
data_dir = os.path.join(save_dir, 'Data')
if not os.path.exists(save_dir): os.makedirs(save_dir)
if not os.path.exists(fig_dir): os.makedirs(fig_dir)
if not os.path.exists(info_dir): os.makedirs(info_dir)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
if not os.path.exists(data_dir): os.makedirs(data_dir)
    
update_rule = sys.argv[2]
special = bool(int(sys.argv[3]))

objective = 'nsgan'
num_iter = int(sys.argv[6])
x0 = None
freq_checkpoint = 250
freq_show = 250
freq_log = 250
n_latent = 100
image_size = 28
batch_size = int(sys.argv[5])
precise = False
regularization = float(sys.argv[4])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = DCGAN_Generator().to(device)
G.apply(DCGAN_weights_init)

D = DCGAN_Discriminator().to(device)
D.apply(DCGAN_weights_init)
    
n_g = sum(p.numel() for p in G.parameters())
n_d = sum(p.numel() for p in D.parameters())

leader_param_index = network_param_index(G)

if special:
    dataset = torch.load(os.path.join(os.getcwd(), 'data', 'MNIST', 'processed', 'special.pt'))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
else:
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),])
    dataset = torchvision.datasets.MNIST(root='./data', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
     
lr_g = 2e-4
lr_d = 2e-4

if update_rule == 'stack':
    gamma_g = 0.99999
    gamma_d = 0.9999999
elif update_rule == 'simgrad':
    gamma_g = 0.9999999
    gamma_d = 0.9999999


m = torch.zeros(n_g, 1).to(device)
v = torch.zeros(n_g, 1).to(device)
beta1 = 0.5
beta2 = 0.999
epsilon = 10**(-8)
opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(beta1, beta2))
sch_D = torch.optim.lr_scheduler.ExponentialLR(opt_D, gamma=gamma_d)

config_names = ['objective', 'update_rule', 'batch_size', 'lr_g', 'lr_d', 
                'gamma_g', 'gamma_d', 'regularization', 'special']
config_choices = [objective, update_rule, batch_size, lr_g, lr_d, 
                  gamma_g, gamma_d, regularization, special]
with open(os.path.join(info_dir, 'config.txt'), "w") as f:
    print('Generator: \n', file=f)
    print(G, file=f)
    print('\n\nDiscriminator: \n', file=f)
    print(D, file=f)
    print('\n', file=f)
    for name, choice in zip(config_names, config_choices):
        print(name + ':', choice, file=f)
        print('\n', file=f)
         
log_data = []
G_latent_show = torch.randn(batch_size, n_latent, 1, 1).to(device)
step = 0

while step <= num_iter:
    
    for i, data in enumerate(dataloader):
        
        if data[0].shape[0] != batch_size: continue
            
        step += 1
        
        G_latent = torch.randn(batch_size, n_latent, 1, 1).to(device)
        real_data = data[0].to(device)

        # G(z) 
        G_generated = G(G_latent)

        # D(x)
        p1 = D(real_data)

        # D(G(z)) 
        p2 = D(G_generated)

        # Compute Loss.
        G_loss, D_loss = objective_function(p1, p2, objective)

        # Obtain Generator Update.
        leader_grad, leader_grad_norm, q_norm, g_norm, x0 = compute_leader_grad(G, D, G_loss, D_loss, regularization, x0, update_rule, 
                                                                                precise=precise, device=device)
        leader_grad = adam_grad(leader_grad, beta1, beta2, epsilon, m, v, step, leader_param_index)
        
        # Optimize Discriminator
        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)
        opt_D.step()
        sch_D.step()
        follower_grad_norm = calc_gradient_norm(D.parameters())
        lr_d_state = opt_D.param_groups[0]['lr']
        
        # Optimize Generator.
        lr_g_state = leader_step(G, leader_grad, lr_g, gamma_g, step)
        
        if step == 1: 
            real_save_data_name = os.path.join(data_dir, 'real_sample')
            np.save(real_save_data_name, real_data.cpu().detach())
            
            real_save_name = os.path.join(fig_dir, 'real_sample')
            mnist_show(real_data.cpu().detach(), disp_num=batch_size, show=show, save=real_save_name)

        if step % freq_log == 0:
            data = log(iter=step, follower_grad_norm=follower_grad_norm, g_norm=g_norm, 
                        q_norm=q_norm, leader_grad_norm=leader_grad_norm, lr_d=lr_d_state, 
                        lr_g=lr_g_state, dis_loss=D_loss.data, gen_loss=G_loss.data)
            log_data.append(data)
            
        if step % freq_show == 0:
            G_generated = G(G_latent_show)
            generator_save_data_name = os.path.join(data_dir, 'generator_step_' + str(step))
            np.save(generator_save_data_name, G_generated.cpu().detach())
            
            generator_save_name = os.path.join(fig_dir, 'generator_step_' + str(step))
            mnist_show(G_generated.cpu().detach(), disp_num=batch_size, show=show, save=generator_save_name)
            
        if step % freq_checkpoint == 0:
            torch.save({'state_gen':G.state_dict(), 'state_dis':D.state_dict()}, 
                         os.path.join(checkpoint_dir, 'checkpoint_step_' + str(step)))
            
            
        if step == num_iter+1: break
