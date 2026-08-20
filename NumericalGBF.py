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
omega = np.linspace(0.01, 1.0, 20)
mass = 0.5

def system(x, Y, M = mass, l = mode, om = omega):

    u_re, u_im, du_re, du_im = Y

    u = u_re + 1j*u_im
    du = du_re + 1j*du_im

    A = x*(1 - x)**2
    B = (1 - x)*(1 - 3*x) - 4*M*1j*om
    C = -(l*(l + 1) - 3*(1 - x))

    if np.abs(A) < 1e-15:
        return np.array([0.0, 0.0, 0.0, 0.0])

    d2u = -(B*du + C*u)/A

    return np.array([du.real, du.imag, d2u.real, d2u.imag])

def robin_BC_h(u0, M, l, om):

    B0 = 1 - 4*M*1j*om
    C0 = -(l*(l + 1) - 3)

    du0 = -C0*u0/B0
    return u0, du0

eps = 1e-8
u0 = complex(1, 0)

print("Initialising solver...")

solutions = {}
GBFs = []

for om in omega:

    u_BC, du_BC = robin_BC_h(u0, mass, mode, om)
    initial_state_Radau = [u_BC.real, u_BC.imag, du_BC.real, du_BC.imag]

    sol = solve_ivp(system, 
                t_span = (eps, 1.0 - 1e-4),
                y0 = initial_state_Radau,
                method = 'Radau',
                args = (mass, mode, om),
                rtol = 1e-10,
                atol = 1e-12
                )

    x_vals = sol.t
    u_sol = sol.y[0] + 1j*sol.y[1]
    du_sol = sol.y[2] + 1j*sol.y[3]

    solutions[om] = (x_vals, u_sol, du_sol)

    x_end = x_vals[-1]
    y = 1.0 - x_end
    u_end = u_sol[-1]
    du_end = du_sol[-1]

    L = mode*(mode + 1)
    Omega = 4*mass*om

    a1 = -1j*L/Omega
    a2 = ((L - 2)*a1 + 3)/(1j*Omega*2.0)

    u1 = 1.0 + a1*y + a2*y**2
    du1_dx = -(a1 + 2.0*a2*y)

    log_deriv =1j*Omega*(1/y**2 + 1/y)
    alpha = (u_end - du_end/log_deriv)/(u1 - du1_dx/log_deriv)
    GBF = 1.0/(np.abs(alpha)**2)
    GBFs.append(GBF)

GBFs = np.array(GBFs)

print("Solver successful. Plotting results...")

omega_last = omega[-1]
x_last, u_last, du_last = solutions[omega_last]

plt.figure(figsize = [6,4])
plt.plot(x_last, u_last.real, label = 'Re(u)')
plt.plot(x_last, u_last.imag, label = 'Im(u)')
plt.plot(x_last, -mode*(mode + 1) + 3*(1 - x_last), label = 'V(x)', linestyle = '--')
plt.xlabel('x', fontsize = 18)
plt.ylabel('u(x)', fontsize = 18)
plt.legend(loc = 'best')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize = [6,4])
plt.plot(omega, GBFs, 'o-', label = f"$l = {int(mode)}$")
plt.xlabel(r'$\omega$', fontsize = 18)
plt.ylabel(r'\Gamma(\omega)', fontsize = 18)
plt.title('Greybody Factor vs Frequency', fontsize = 20)
plt.legend(loc = 'lower right')
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.grid()
plt.show()