import torch as t
import numpy as np 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import random
from scipy.integrate import solve_ivp
from Functions import *

x_test = np.linspace(0, 1, 1000)

def dense_grid(n_points, x_peak, peak_width, peak_intensity, bound_width, bound_intensity):

    ref_x = np.linspace(0, 1, 25000)
    weights = 1.0 + peak_intensity*np.exp(-(ref_x - x_peak)**2/(2*peak_width**2)) 
    weights += bound_intensity*np.exp(-(ref_x - 1)**2/(2*bound_width**2)) 
    weights += bound_intensity*np.exp(-(ref_x - 0)**2/(2*bound_width**2)) 
    
    cdf = np.cumsum(weights)
    cdf /= cdf[-1] 

    uniform_steps = np.linspace(0.0 + 1e-8, 1 - 1e-2, n_points)
    clustered_x = np.interp(uniform_steps, cdf, ref_x)
    clustered_x[0] = 0
    clustered_x[-1] = 1
    
    return clustered_x

M = 1.0/2
l = 2
x_peak = Vpm_peak(M = 1/2, l = 2, parity = "axial", coord = 'x')

N = 100000

x_grid = dense_grid(N, x_peak, peak_intensity = 13, peak_width = 0.08, 
                    bound_intensity = 16, bound_width = 0.04)


#PINN architecture
class adaptive_tanh(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.beta = nn.Parameter(t.zeros(1, num_features))

    def forward(self, x):
        return (1 + self.beta*x)*t.tanh(x)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers=2):
        super().__init__() 
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        self.act_input = adaptive_tanh(hidden_channels)

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.activations.append(adaptive_tanh(hidden_channels))

        self.output_layer = nn.Linear(hidden_channels, out_channels) 

    def forward(self, x: t.tensor):
        x = self.input_layer(x)
        x = self.act_input(x)

        for layer, act in zip(self.hidden_layers, self.activations):
            x = layer(x)
            x = act(x)

        x = self.output_layer(x) 
        return x 

#Set up
DTYPE = t.float32
NP_DTYPE = np.float32 if DTYPE == t.float32 else np.float64
mass = 0.5
mode = 2
x_horizon = t.tensor(0., requires_grad = True, dtype = DTYPE).view(-1, 1)
x_infinity = t.tensor(1., requires_grad = True, dtype = DTYPE).view(-1, 1)
x_tensor = t.tensor(x_grid, requires_grad = True, dtype = DTYPE).view(-1, 1)

#Define functions
def grads(y, x):
        dy = t.autograd.grad(y, x, t.ones_like(y), create_graph = True)[0]
        d2y = t.autograd.grad(dy, x, t.ones_like(dy), create_graph = True)[0]
        return dy, d2y

def A(x):
    A = x*(1 - x)**2
    return A

def B(x, rho_re, rho_im, M):
        B_re = (4*rho_re + 3)*x**2 - 4*(2*rho_re + 1)*x + 2*rho_re + 1
        B_im = (4*rho_im)*x**2 - 4*(2*rho_im)*x + 2*rho_im
        return B_re, B_im

def C(x, rho_re, rho_im, M, l):
    C_re = -l*(l + 1) - 3*(x - 1) + 4*rho_re*x - 8*(rho_re**2 - rho_im**2) + \
    4*(rho_re**2 - rho_im**2)*x - 4*rho_re
    C_im = 4*rho_im*x - 16*rho_re*rho_im + 8*rho_re*rho_im*x - 4*rho_im
    return C_re, C_im

def annealing(epoch, total_epochs):
    if epoch < 0.2*total_epochs:
        BC = 100.0
        NT = 100.0
        ODE = 0.01
    
    else:
        BC = 10.0
        NT = 10.0
        ODE = 1.0
    
    return [BC, NT, BC, ODE]

loss_h = []
loss_nt = []
loss_inf = []
loss_ODE = []

learning_rate = 1e-3
model = Model(1, 2, 32, num_hidden_layers = 2).to(dtype = DTYPE)
# rho_real = nn.Parameter(t.tensor([-0.08], requires_grad = True, dtype = DTYPE))
# rho_imag = nn.Parameter(t.tensor([-0.37], requires_grad = True, dtype = DTYPE))
rho_real = t.tensor(-0.089, requires_grad = False, dtype = DTYPE)
rho_imag = t.tensor(-0.3737, requires_grad = False, dtype = DTYPE)
# optimiser = optim.Adam([
#     {'params': model.parameters(), 'lr': 1e-3},
#     {'params': [rho_real, rho_imag], 'lr': 1e-2}])

optimiser = optim.Adam([{'params': model.parameters(), 'lr': 1e-3}])

number_iterations = 5000
loss_total = []
rho_real_array = []
rho_imag_array = []
for epoch in range(number_iterations):

    loss_weights = annealing(epoch, number_iterations)

    optimiser.zero_grad()
    rho_real_array.append(rho_real.item())
    rho_imag_array.append(rho_imag.item())

    #Horizon BC
    w_h = model(x_horizon)
    w_h_re, w_h_im = w_h[:, 0:1], w_h[:, 1:2]
    dw_h_re, d2w_h_re = grads(w_h_re, x_horizon)
    dw_h_im, d2w_h_im = grads(w_h_im, x_horizon)

    B_h_re, B_h_im = B(x_horizon, rho_real, rho_imag, mass)
    C_h_re, C_h_im = C(x_horizon, rho_real, rho_imag, mass, mode)

    loss_horizon_re = (B_h_re*dw_h_re - B_h_im*dw_h_im +
                        C_h_re*w_h_re - C_h_im*w_h_im)
    loss_horizon_im = (B_h_im*dw_h_re + B_h_re*dw_h_im + 
                    C_h_im*w_h_re + C_h_re*w_h_im)
    loss_horizon = loss_horizon_re**2 + loss_horizon_im**2
    loss_nontrivial = (1.0 - w_h_re)**2 + (0.0 - w_h_im)**2

    #Infinity BC
    w_inf = model(x_infinity)
    w_inf_re, w_inf_im = w_inf[:, 0:1], w_inf[:, 1:2]
    dw_inf_re, d2w_inf_re = grads(w_inf_re, x_infinity)
    dw_inf_im, d2w_inf_im = grads(w_inf_im, x_infinity)

    B_inf_re, B_inf_im = B(x_infinity, rho_real, rho_imag, mass)
    C_inf_re, C_inf_im = C(x_infinity, rho_real, rho_imag, mass, mode)

    loss_infinity_re = (B_inf_re*dw_inf_re - B_inf_im*dw_inf_im + 
                        C_inf_re*w_inf_re - C_inf_im*w_inf_im)
    loss_infinity_im = (B_inf_im*dw_inf_re + B_inf_re*dw_inf_im + 
                        C_inf_im*w_inf_re + C_inf_re*w_inf_im)
    loss_infinity = loss_infinity_re**2 + loss_infinity_im**2

    #Physics loss
    w_nn = model(x_tensor)
    w_nn_re, w_nn_im = w_nn[:, 0:1], w_nn[:, 1:2]
    # loss_nontrivial = -(w_nn_re**2 + w_nn_im**2).mean()
    dw_nn_re, d2w_nn_re = grads(w_nn_re, x_tensor)
    dw_nn_im, d2w_nn_im = grads(w_nn_im, x_tensor)

    A_ = A(x_tensor)
    B_re, B_im = B(x_tensor, rho_real, rho_imag, mass)
    C_re, C_im = C(x_tensor, rho_real, rho_imag, mass, mode)

    loss_ode_re = (A_*d2w_nn_re + B_re*dw_nn_re - B_im*dw_nn_im + C_re*w_nn_re - C_im*w_nn_im)
    loss_ode_im = (A_*d2w_nn_im + B_im*dw_nn_re + B_re*dw_nn_im + C_im*w_nn_re + C_re*w_nn_im)
    loss_ode = t.mean(loss_ode_re**2 + loss_ode_im**2)

    loss = (loss_weights[0]*t.mean(loss_horizon) + loss_weights[1]*t.mean(loss_nontrivial) + 
            loss_weights[2]*t.mean(loss_infinity) + loss_weights[3]*loss_ode)

    loss.backward()
    optimiser.step()

    with t.no_grad():
        loss_h.append(loss_horizon.item())
        loss_nt.append(loss_nontrivial.item())
        loss_inf.append(loss_infinity.item())
        loss_ODE.append(loss_ode.item())
        loss_total.append(loss.item())

    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"""Epoch: {epoch + 1} / {number_iterations}. Total scaled loss: {loss.item():.4e}, 
                    ODE(0) loss: {loss_horizon.item():.4e}, 
                    ODE(1) loss: {loss_infinity.item():.4e}, 
                    Non-trivial solution loss: {loss_nontrivial.item():.4e},
                    Physics loss: {loss_ode.item():.4e},
                    Current value of rho: {rho_real.item():.5f} + {rho_imag.item():.5f}i""")
        print("-"*30)

    if (epoch + 1) % 1000 == 0:
        plt.figure()
        plt.semilogy(loss_total, label = 'Total')
        plt.plot(loss_h, label = 'Horizon')
        plt.plot(loss_inf, label = 'Infinity')
        plt.plot(loss_ODE, label ='ODE')
        plt.plot(np.abs(loss_nt), label = 'Non-trivial solution')
        plt.ylabel('Loss', fontsize = 20)
        plt.xlabel('Epoch', fontsize = 20)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        plt.figure()
        plt.plot(x_grid, loss_ode_re.detach().numpy(), color = 'blue', label = r'$\Re (Res_{ODE})$')
        plt.plot(x_grid, loss_ode_im.detach().numpy(), color = 'green', label = r'$\Im (Res_{ODE})$')
        plt.plot(x_grid, w_nn_re.detach().numpy(), color = 'orange', label = r'$\Re (w_{NN})$')
        plt.plot(x_grid, w_nn_im.detach().numpy(), color = 'red', label = r'$\Im (w_{NN})$')
        plt.xlabel('x', fontsize = 20)
        plt.ylabel('Residual', fontsize = 20)
        plt.title('Residual over space', fontsize = 22)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
    
losses = np.array([loss_h, loss_nt, loss_inf, loss_ODE, loss_total])
rho_real_array = np.array([rho_real_array])
rho_imag_array = np.array([rho_imag_array])