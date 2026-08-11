import torch as t
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributions as dist
import os
import argparse
from Functions import *
from matplotlib import rc

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "text.usetex": False
})

j = complex(0, 1)

parser = argparse.ArgumentParser(description = "Train PINN for specific mode l")
parser.add_argument('--mode', type = int, required = True, help = 'The value of l (mode)')
parser.add_argument('--omega', type = float, required = True, help = 'Incident wave frequency')
# parser.add_argument('--delta', type = float, default = 0.1, help = 'Range of random perturbation away from true QNM')

args = parser.parse_args()
mode = args.mode
omega = args.omega
mass = 0.5
x_max = 0.95
rstar_max = r_to_rstar(x_to_r(x_max, mass), mass)
alpha_init = complex(1, 0)
beta_init = complex(0, 1)

# omega_dict = {
#     2: complex(0.74734, -0.17792),
#     3: complex(1.19889, -0.18541),
#     4: complex(1.61835, -0.18832),
#     5: complex(2.02458, -0.18974),
#     6: complex(2.42402, -0.19053)
# }

# if mode not in omega_dict:
#     raise ValueError(f"Mode l={mode} is not supported. Choose from {list(omega_dict.keys())}.")

base_path = f'./Output/l{mode}/omega{omega}'

out_dir = os.path.join(base_path, 'NNOutput')
loss_dir = os.path.join(base_path, 'Loss')
os.makedirs(base_path, exist_ok=True)
os.makedirs(out_dir, exist_ok = True)
os.makedirs(loss_dir, exist_ok = True)

device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#PINN architecture
class adaptive_tanh(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.beta = nn.Parameter(t.zeros(1, num_features))

    def forward(self, x):
        return (1 + self.beta*x)*t.tanh(x)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_hidden_layers=2, alpha_in = alpha_init, beta_in = beta_init):
        super().__init__() 
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        self.act_input = adaptive_tanh(hidden_channels)

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.activations.append(adaptive_tanh(hidden_channels))

        self.output_layer = nn.Linear(hidden_channels, out_channels) 

        self.alpha_re = nn.Parameter(t.tensor(alpha_init.real, dtype = DTYPE), requires_grad = False)
        self.alpha_im = nn.Parameter(t.tensor(alpha_init.imag, dtype = DTYPE), requires_grad = False)
        self.beta_re = nn.Parameter(t.tensor(beta_init.real, dtype = DTYPE), requires_grad = False)
        self.beta_im = nn.Parameter(t.tensor(beta_init.imag, dtype = DTYPE), requires_grad = False)

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

#Define functions
def grads(y, x):
        dy = t.autograd.grad(y, x, t.ones_like(y), create_graph = True)[0]
        d2y = t.autograd.grad(dy, x, t.ones_like(dy), create_graph = True)[0]
        return dy, d2y

def first_grad(y, x):
    dy = t.autograd.grad(y, x, t.ones_like(y), create_graph = True)[0]
    return dy

def A(x):
    A = x*(1 - x)**2
    return A

def B(x, M, omega):
    real = (1 - x) * (1 - 3*x)
    imag = -4 * M * omega * t.ones_like(x)
    return t.complex(real, imag)
        
def C(x, l):
    return l*(l  + 1) - 3*(1 - x)

def g(x, M):
    return x*(1 - x)**2/(2*M)

def reciprocal_z(z):
    z_re, z_im = z.real, z.imag
    mod_sq_z = z_re**2 + z_im**2
    conjugate = complex(z_re, -z_im)
    return conjugate/mod_sq_z

def annealing(epoch, total_epochs):
    if epoch <= 0.1*total_epochs:
        BC = 100.0
        AMPLITUDE = 100.0
        UMAX = 10.0
        UMAX_DERIV = 10.0
        ODE = 0.1
        WRON = 0.1

    elif 0.1*total_epochs < epoch < 0.6*total_epochs:
        BC = 100.0
        AMPLITUDE = 100.0
        UMAX = (50.0 - 10.0)/(0.6 - 0.1)*(epoch/total_epochs) + 2
        UMAX_DERIV = (50.0 - 10.0)/(0.6 - 0.1)*(epoch/total_epochs) + 2
        ODE = (10.0 - 0.1)/(0.6 - 0.1)*(epoch/total_epochs) - 1.88
        WRON = 0.1

    elif epoch >= 0.6*total_epochs:
        BC = 100.0
        AMPLITUDE = 100.0
        UMAX = 50.0
        UMAX_DERIV = 50.0
        ODE = 10
        WRON = (100.0 - 0.1)/(1.0 - 0.6)*(epoch/total_epochs) - 149.75
    
    return [BC, AMPLITUDE, UMAX, UMAX_DERIV, ODE, WRON]

def compute_loss(model, x_tensor, weights, mass, mode, omega):
    """weights: [weight_horizon, weight_amplitude, weight_umax, 
    weight_umax_deriv, weight_ODE, weight_wronskian]
    returns: Re(w_nn), Im(w_nn), total loss, horizon loss, amplitude loss, 
    u boundary loss, u' boundary loss, ODE loss, Re(ODE loss), Im(ODE loss), wronskian loss"""

    #Boundary tensors
    x_horizon = t.tensor(0., requires_grad = True, dtype = DTYPE, device = device).view(-1, 1)  
    x_max_tensor = t.tensor(x_max, requires_grad = True, dtype = DTYPE, device = device).view(-1, 1) 

    #Horizon loss
    u_h = model(x_horizon)
    u_h_re, u_h_im = u_h[:, 0:1], u_h[:, 1:2]
    du_h_re = first_grad(u_h_re, x_horizon)
    du_h_im = first_grad(u_h_im, x_horizon)

    B_h = B(x_horizon, mass, omega)
    B_h_re, B_h_im = B_h.real, B_h.imag
    C_h = C(x_horizon, mode)

    #Robin boundary condition
    loss_horizon_re = B_h_re*du_h_re - B_h_im*du_h_im + C_h*u_h_re
    loss_horizon_im = B_h_im*du_h_re + B_h_re*du_h_im + C_h*u_h_im
    loss_horizon = t.mean(loss_horizon_re**2 + loss_horizon_im**2)

    #Horizon amplitude normalisation
    loss_amplitude = t.mean((1.0 - u_h_re)**2 + (0.0 - u_h_im)**2)

    #x_max BC loss terms
    u_max = model(x_max_tensor)
    u_max_re, u_max_im = u_max[:, 0:1], u_max[:, 1:2]
    du_max_re = first_grad(u_max_re, x_max_tensor)
    du_max_im = first_grad(u_max_im, x_max_tensor)

    #u boundary term: (u_NN - u_analytical)|_x_max
    boundary_real = u_max_re - (model.alpha_re + model.beta_re*t.cos(2*omega*rstar_max)
                        - model.beta_im*t.sin(2*omega*rstar_max))
    boundary_imag = u_max_im - (model.alpha_im + model.beta_im*t.cos(2*omega*rstar_max)
                        + model.beta_re*t.sin(2*omega*rstar_max))
    loss_u_max = t.mean(boundary_real**2 + boundary_imag**2)

    #u derivative boundary term: (u'_NN - u'_analytical)|_x_max
    deriv_boundary_real = du_max_re + 2*omega*(model.beta_im*t.cos(2*omega*rstar_max) 
                                        + model.beta_re*t.sin(2*omega*rstar_max))/g(x_max, mass)
    deriv_boundary_imag = du_max_im - 2*omega*(model.beta_re*t.cos(2*omega*rstar_max)
                                        - model.beta_im*t.sin(2*omega*rstar_max))/g(x_max, mass)
    loss_deriv_u_max = t.mean(deriv_boundary_real**2 + deriv_boundary_imag**2)

    #Physics loss
    u_nn = model(x_tensor)
    u_nn_re, u_nn_im = u_nn[:, 0:1], u_nn[:, 1:2]
    du_nn_re, d2u_nn_re = grads(u_nn_re, x_tensor)
    du_nn_im, d2u_nn_im = grads(u_nn_im, x_tensor)

    A_ = A(x_tensor)
    B_ = B(x_tensor, mass, omega)
    B_re, B_im = B_.real, B_.imag
    C_ = C(x_tensor, mode)

    loss_ode_re = (A_*d2u_nn_re + B_re*du_nn_re - B_im*du_nn_im + C_*u_nn_re)
    loss_ode_im = (A_*d2u_nn_im + B_im*du_nn_re + B_re*du_nn_im + C_*u_nn_im)
    loss_ode = t.mean(loss_ode_re**2 + loss_ode_im**2)

    #Wronskian/Probability flux conservation loss
    loss_wronskian = (1 - (1 + model.beta_re**2 + model.beta_im**2)/(model.alpha_re**2 + model.alpha_im**2 + 1e-8))**2

    total_loss = (weights[0]*loss_horizon + weights[1]*loss_amplitude + weights[2]*loss_u_max
                + weights[3]*loss_deriv_u_max + weights[4]*loss_ode + weights[5]*loss_wronskian)
    
    return u_nn_re, u_nn_im, total_loss, loss_horizon, loss_amplitude, loss_u_max, loss_deriv_u_max, loss_ode, loss_ode_re, loss_ode_im, loss_wronskian

# omega = omega_dict[mode]
# rho = -1j*omega
# rho_perturbed = rho + complex(np.random.uniform(-delta, delta), np.random.uniform(-delta, delta))

print('-'*30)
print(f'Starting training for l = {mode} with omega = {omega}')
# initial_dist = np.abs(rho_perturbed - rho)
# print(f"Initial rho: {rho_perturbed.real:.5f} + {rho_perturbed.imag:.5f}i")
# print(f"Target rho: {rho.real:.5f} + {rho.imag:.5f}i")
# print(f"Initial distance from target: {initial_dist:.5f}")
print('-'*30)

loss_h = []
loss_amp = []
loss_umax = []
loss_umax_deriv = []
loss_ODE = []
loss_wronskian = []
loss_total = []
alpha_real_array = []
alpha_imag_array = []
beta_real_array = []
beta_imag_array = []

N_points = 10000
learning_rate = 1e-3
model = Model(1, 2, 32, num_hidden_layers = 3).to(device = device, dtype = DTYPE)

adam_parameters = [p for p in model.parameters() if p is not model.alpha_re and p is not model.alpha_im 
                and p is not model.beta_re and p is not model.beta_im]

optimiser = optim.Adam(adam_parameters, lr = learning_rate)

#Define sampling distribution once
beta_dist = dist.Beta(t.tensor([0.5], device = device), t.tensor([0.5], device = device))

Adam_iterations = 18000
for epoch in range(Adam_iterations):
    optimiser.zero_grad()
    N_uniform = int(0.65*N_points)
    x_uniform = x_max*t.rand((N_uniform, 1), dtype = DTYPE, device = device)

    N_edges = N_points - N_uniform
    x_edges = x_max*beta_dist.sample((N_edges,)).view(-1, 1).to(device = device, dtype = DTYPE)

    x_tensor = t.cat([x_uniform, x_edges], dim = 0)
    x_tensor.requires_grad_(True)

    if epoch == int(0.1*Adam_iterations):
        print("Unfreezing coefficients...")
        model.alpha_re.requires_grad = True
        model.alpha_im.requires_grad = True
        model.beta_re.requires_grad = True
        model.beta_im.requires_grad = True
        optimiser.add_param_group({'params': [model.alpha_re, model.alpha_im, 
                                            model.beta_re, model.beta_im], 'lr': 1e-4})

    loss_weights = annealing(epoch, Adam_iterations)
    Re_u_nn, Im_u_nn, loss, l_h, l_amp, l_umax, l_umax_deriv, l_ode, l_ode_re, \
        l_ode_im, l_wron = compute_loss(model, x_tensor, loss_weights, mass, mode, omega)

    loss.backward()
    optimiser.step()

    alpha_real_array.append(model.alpha_re.item())
    alpha_imag_array.append(model.alpha_im.item())
    beta_real_array.append(model.beta_re.item())
    beta_imag_array.append(model.beta_im.item())

    with t.no_grad():
        loss_h.append(l_h.item())
        loss_amp.append(l_amp.item())
        loss_umax.append(l_umax.item())
        loss_umax_deriv.append(l_umax_deriv.item())
        loss_ODE.append(l_ode.item())
        loss_wronskian.append(l_wron.item())
        loss_total.append(loss.item())

    if (epoch + 1) % 500 == 0 or epoch == 0 or epoch == (Adam_iterations - 1):
        T = reciprocal_z(complex(model.alpha_re.item(), model.alpha_im.item()))
        R = complex(model.beta_re.item(), model.beta_im.item())*T
        wronskian = np.abs(T)**2 + np.abs(R)**2
        print(f"""Epoch: {epoch + 1} / {Adam_iterations}. Total scaled loss: {loss.item():.4e}, 
                    ODE(0) loss: {l_h.item():.4e}, 
                    Amplitude loss: {l_amp.item():.4e}, 
                    u(x_max) loss: {l_umax.item():.4e},
                    u'(x_max) loss: {l_umax_deriv.item():.4e},
                    Physics loss: {l_ode.item():.4e},
                    Wronskian loss: {l_wron.item():.4e},
                    Current value of alpha: {model.alpha_re.item():.5f} + {model.alpha_im.item():.5f}i,
                    Current value of beta: {model.beta_re.item():.5f} + {model.beta_im.item():.5f}i,
                    Current value of T: {T.real:.5f} + {T.imag:.5f}i,
                    Current value of R: {R.real:.5f} + {R.imag:.5f}i,
                    Current value of |T|^2 + |R|^2: {wronskian:.5f} .
                    """)
        print("-"*30)

    if (epoch + 1) % 1000 == 0:
        x_np = x_tensor.cpu().detach().numpy().flatten()
        idx = np.argsort(x_np)

        res_re_plot = l_ode_re.cpu().detach().numpy().flatten()
        res_im_plot = l_ode_im.cpu().detach().numpy().flatten()
        u_re_plot = Re_u_nn.cpu().detach().numpy().flatten()
        u_im_plot = Im_u_nn.cpu().detach().numpy().flatten()

        plt.figure()
        plt.plot(x_np[idx], res_re_plot[idx], color = 'blue', label = r'$\Re (Res_{ODE})$')
        plt.plot(x_np[idx], res_im_plot[idx], color = 'green', label = r'$\Im (Res_{ODE})$')
        plt.plot(x_np[idx], u_re_plot[idx], color = 'orange', label = r'$\Re (u_{NN})$')
        plt.plot(x_np[idx], u_im_plot[idx], color = 'red', label = r'$\Im (u_{NN})$')
        plt.xlabel('x', fontsize = 25)
        plt.ylabel('Output', fontsize = 25)
        plt.grid()
        plt.legend(fontsize = 25)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/Output_Epoch_{epoch + 1}.png', format = 'png')
        plt.close()

        plt.figure()
        plt.plot(loss_total, label = 'Total')
        plt.plot(loss_h, label = 'Horizon')
        plt.plot(loss_amp, label = 'Amplitude')
        plt.plot(loss_umax, label = 'u(x_max)')
        plt.plot(loss_umax_deriv, label = "u'(x_max)")
        plt.plot(loss_ODE, label ='ODE')
        plt.plot(loss_wronskian, label = 'Wronskian')
        plt.yscale('log')
        plt.ylabel('Loss', fontsize = 25)
        plt.xlabel('Epoch', fontsize = 25)
        plt.legend(fontsize = 25)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{loss_dir}/Loss_Epoch_{epoch + 1}.png', format = 'png')
        plt.close()

print("Adam training complete. Switching to L-BFGS:")

model.alpha_re.requires_grad = True
model.alpha_im.requires_grad = True
model.beta_re.requires_grad = True
model.beta_im.requires_grad = True

lbfgs_optimiser = t.optim.LBFGS(model.parameters(), lr = 1.0, max_iter = 20,
            history_size = 50, line_search_fn = 'strong_wolfe')

lbfgs_iterations = 1000
lbfgs_weights = annealing(Adam_iterations, Adam_iterations)

N_uniform = int(0.65*N_points)
x_uniform = x_max*t.rand((N_uniform, 1), dtype = DTYPE, device = device)

N_edges = N_points - N_uniform
x_edges = x_max*beta_dist.sample((N_edges,)).view(-1, 1).to(device = device, dtype = DTYPE)

x_tensor_lbfgs = t.cat([x_uniform, x_edges], dim = 0)
x_tensor_lbfgs.requires_grad_(True)

for epoch in range(lbfgs_iterations):
    info = {'total': 0, 'h': 0, 'amp': 0, 'umax': 0, 'umaxderiv':0, 'ode':0, 'wronskian': 0}
    plot_data = {}

    def closure():
        lbfgs_optimiser.zero_grad()

        Re_u_nn, Im_u_nn, loss, l_h, l_amp, l_umax, l_umax_deriv, l_ode, l_ode_re, l_ode_im, l_wron = compute_loss(model, x_tensor_lbfgs, lbfgs_weights, mass, mode, omega)       
        loss.backward()

        info.update({'total': loss.item(), 'h': l_h.item(), 'amp': l_amp.item(),
                    'umax': l_umax.item(), 'umaxderiv': l_umax_deriv.item(), 'ode': l_ode.item(), 'wronskian': l_wron.item()})

        plot_data['x'] = x_tensor_lbfgs.cpu().detach().numpy()
        plot_data['re_w'] = Re_u_nn.cpu().detach().numpy()
        plot_data['im_w'] = Im_u_nn.cpu().detach().numpy()
        plot_data['res_re'] = l_ode_re.cpu().detach().numpy()
        plot_data['res_im'] = l_ode_im.cpu().detach().numpy()

        return loss

    lbfgs_optimiser.step(closure)

    loss_h.append(info['h'])
    loss_amp.append(info['amp'])
    loss_umax.append(info['umax'])
    loss_umax_deriv.append(info['umaxderiv'])
    loss_ODE.append(info['ode'])
    loss_wronskian.append(info['wronskian'])
    loss_total.append(info['total'])

    alpha_real_array.append(model.alpha_re.item())
    alpha_imag_array.append(model.alpha_im.item())
    beta_real_array.append(model.beta_re.item())
    beta_imag_array.append(model.beta_im.item())

    if (epoch + 1) % 100 == 0:
        T = reciprocal_z(complex(model.alpha_re.item(), model.alpha_im.item()))
        R = complex(model.beta_re.item(), model.beta_im.item())*T
        wronskian = np.abs(T)**2 + np.abs(R)**2
        print(f"""L-BFGS Epoch: {epoch + 1} / {lbfgs_iterations}. Total scaled loss: {info['total']:.4e}, 
                    ODE(0) loss: {info['h']:.4e}, 
                    Amplitude loss: {info['amp']:.4e}, 
                    u(x_max) loss: {info['umax']:.4e},
                    u'(x_max) loss: {info['umaxderiv']:.4e},
                    Physics loss: {info['ode']:.4e},
                    Wronskian loss: {info['wronskian']:.4e},
                    Current value of alpha: {model.alpha_re.item():.5f} + {model.alpha_im.item():.5f}i,
                    Current value of beta: {model.beta_re.item():.5f} + {model.beta_im.item():.5f}i,
                    Current value of T: {T.real:.5f} + {T.imag:.5f}i,
                    Current value of R: {R.real:.5f} + {R.imag:.5f}i,
                    Current value of |T|^2 + |R|^2: {wronskian:.5f} .
                    """)
        print("-"*30)


    if epoch == (lbfgs_iterations - 1):
        print(f"""Training complete for l = {mode} with omega = {omega}. Total scaled loss: {info['total']:.4e}, 
                    ODE(0) loss: {info['h']:.4e}, 
                    Amplitude loss: {info['amp']:.4e}, 
                    u(x_max) loss: {info['umax']:.4e},
                    u'(x_max) loss: {info['umaxderiv']:.4e},
                    Physics loss: {info['ode']:.4e},
                    Wronskian loss: {info['wronskian']:.4e},
                    Current value of alpha: {model.alpha_re.item():.5f} + {model.alpha_im.item():.5f}i,
                    Current value of beta: {model.beta_re.item():.5f} + {model.beta_im.item():.5f}i,
                    Current value of T: {T.real:.5f} + {T.imag:.5f}i,
                    Current value of R: {R.real:.5f} + {R.imag:.5f}i,
                    Current value of |T|^2 + |R|^2: {wronskian:.5f} .
                    """)

    if (epoch + 1) % 100 == 0 or epoch == (lbfgs_iterations - 1):
        x_plot = plot_data['x'].flatten()
        idx = np.argsort(x_plot)

        plt.figure()
        plt.plot(x_plot[idx], plot_data['res_re'].flatten()[idx], color = 'blue', label = r'$\Re (Res_{ODE})$')
        plt.plot(x_plot[idx], plot_data['res_im'].flatten()[idx], color = 'green', label = r'$\Im (Res_{ODE})$')
        plt.plot(x_plot[idx], plot_data['re_w'].flatten()[idx], color = 'orange', label = r'$\Re (w_{NN})$')
        plt.plot(x_plot[idx], plot_data['im_w'].flatten()[idx], color = 'red', label = r'$\Im (w_{NN})$')
        plt.xlabel('x', fontsize = 25)
        plt.ylabel('Residual', fontsize = 25)
        plt.grid()
        plt.legend(fontsize = 25)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/Output_Epoch_{epoch + 1 + Adam_iterations}.png', format = 'png')
        plt.close()
    
        plt.figure()
        plt.plot(loss_total, label = 'Total')
        plt.plot(loss_h, label = 'Horizon')
        plt.plot(loss_amp, label = 'Amplitude')
        plt.plot(loss_umax, label = 'u(x_max)')
        plt.plot(loss_umax_deriv, label = "u'(x_max)")
        plt.plot(loss_ODE, label ='ODE')
        plt.plot(loss_wronskian, label = 'Wronskian')
        plt.yscale('log')
        plt.ylabel('Loss', fontsize = 25)
        plt.xlabel('Epoch', fontsize = 25)
        plt.legend(fontsize = 25)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{loss_dir}/Loss_Epoch_{epoch + 1 + Adam_iterations}.png', format = 'png')
        plt.close()

alpha_real_array = np.array(alpha_real_array)
alpha_imag_array = np.array(alpha_imag_array)
beta_real_array = np.array(beta_real_array)
beta_imag_array = np.array(beta_imag_array)

plt.figure(figsize = [7, 7])
plt.plot(alpha_real_array, alpha_imag_array, 'r--', alpha = 0.9)
plt.scatter(alpha_init.real, alpha_init.imag, color='blue', label = f'Initial: {alpha_init.real:.5f} + {alpha_init.imag:.5f}i')
plt.scatter(model.alpha_re.item(), model.alpha_im.item(), color='green', label = f'Final: {model.alpha_re.item():.5f} + {model.alpha_im.item():.5f}i')
plt.xlabel(r'$\Re(\alpha)$', fontsize = 18)
plt.ylabel(r'$\Im(\alpha)$', fontsize = 18)
plt.title(f'l = {mode}, omega = {omega}', fontsize = 18)
plt.tight_layout()
plt.grid()
plt.legend()
plt.savefig(f'{base_path}/alpha_convergence.png', format = 'png')
plt.close()

plt.figure(figsize = [7, 7])
plt.plot(beta_real_array, beta_imag_array, 'r--', alpha = 0.9)
plt.scatter(beta_init.real, beta_init.imag, color='blue', label = f'Initial: {beta_init.real:.5f} + {beta_init.imag:.5f}i')
plt.scatter(model.beta_re.item(), model.beta_im.item(), color='green', label = f'Final: {model.beta_re.item():.5f} + {model.beta_im.item():.5f}i')
plt.xlabel(r'$\Re(\beta)$', fontsize = 18)
plt.ylabel(r'$\Im(\beta)$', fontsize = 18)
plt.title(f'l = {mode}, omega = {omega}', fontsize = 18)
plt.tight_layout()
plt.grid()
plt.legend()
plt.savefig(f'{base_path}/beta_convergence.png', format = 'png')
plt.close()

results = {
    'model_state_dict': model.state_dict(),
    'alpha_history': {'re': alpha_real_array, 'im': alpha_imag_array},
    'beta_history': {'re': beta_real_array, 'im': beta_imag_array},
    'loss_history': {'total': loss_total, 'h': loss_h, 'amp': loss_amp, 'umax': loss_umax, 'umax_deriv': loss_umax_deriv, 'ode': loss_ODE, 'wronskian': loss_wronskian},
    'final_alpha': [model.alpha_re.item(), model.alpha_im.item()],
    'final_beta': [model.beta_re.item(), model.beta_im.item()]
    }

checkpoint_path = os.path.join(base_path,f'pinn_checkpoint_qnm_l{mode}_omega{omega}.pth') 
t.save(results, checkpoint_path)
print(f'Training complete. Checkpoint saved to {checkpoint_path}')
print(f"l = {mode} mode with omega = {omega} completed successfully.")