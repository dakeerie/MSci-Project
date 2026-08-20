import torch as t
import numpy as np 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import torch.distributions as dist
import matplotlib.pyplot as plt 
import random
from scipy.integrate import solve_ivp
from Functions import *

j = complex(0, 1)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": "" # Clear the helvet and sansmath packages
})

j = complex(0, 1)

mode = 2
omega = 0.1
mass = 0.5

def system(x, Y, M = mass, l = mode, om = omega):

    u_re, u_im, du_re, du_im = Y

    u = u_re + 1j*u_im
    du = du_re + 1j*du_im

    A = x*(1 - x)**2
    B = (1 - x)*(1 - 3*x) - 4*M*1j*om
    C = l*(l + 1) - 3*(1 - x)

    if np.abs(A) < 1e-15:
        return np.array([0.0, 0.0, 0.0, 0.0])

    d2u = -(B*du + C*u)/A

    return np.array([du.real, du.imag, d2u.real, d2u.imag])

def robin_BC_h(u0, M = mass, l = mode, om = omega):

    B0 = 1 - 4*M*1j*om
    C0 = l*(l + 1) - 3

    du0 = -C0*u0/B0
    return u0, du0


eps = 1e-8
u0 = complex(1, 0)
BC = robin_BC_h(u0)

initial_state_Radau = [BC[0].real, BC[0].imag, BC[1].real, BC[1].imag]
print("Initialising solver...")
sol = solve_ivp(system, 
                t_span = (eps, 1.0 - 1e-4),
                y0 = initial_state_Radau,
                method = 'Radau',
                args = (mass, mode, omega),
                rtol = 1e-10,
                atol = 1e-12
                )

x_vals = sol.t
u_sol = sol.y[0] + 1j*sol.y[1]
du_sol = sol.y[2] + 1j*sol.y[3]

print("Solver successful. Plotting results...")

plt.figure(figsize = [6,4])
plt.plot(x_vals, u_sol.real, label = 'Re(u)')
plt.plot(x_vals, u_sol.imag, label = 'Im(u)')
plt.plot(x_vals, mode*(mode + 1) - 3*(1 - x_vals), label = 'V(x)')
plt.xlabel('x', fontsize = 18)
plt.ylabel('u(x)', fontsize = 18)
plt.legend(loc = 'best')
plt.grid()
plt.show()