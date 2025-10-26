import numpy as np
import matplotlib.pyplot as plt

def axial(r, M, l):
    V_axial = (1-2*M/r)*(l*(l+1)/r**2 - 6*M/r**3)
    return V_axial

def polar(r, M, l):
    n = 1/2*(l-1)*(l+2)
    mode = n**2*(n+1)*r**3 + 3*M*n**2*r**2 + 9*M**2*n*r + 9*M**3
    coeff = 2*(1-2*M/r)/(r**3*(n*r + 3*M)**2)
    V_polar = coeff*mode
    return V_polar

l_array = [2, 3, 4]
M = 1
eps = 1e-3
r = np.linspace(2*M + eps, 10*M, 1000)
colours = ['red', 'green', 'blue']

print("WORK ON LEGEND")

plt.figure(figsize = [6, 6])
for i in range(len(l_array)):
    polar_V = polar(r, M, l_array[i])
    axial_V = axial(r, M, l_array[i])
    plt.plot(r/M, polar_V, color = colours[i], label = f'Polar potential, l = {l_array[i]}')
    plt.plot(r/M, axial_V, color = colours[i], label = f'Axial potential, l = {l_array[i]}', linestyle = '--')
plt.xlabel('r/M', fontsize = 15)
plt.ylabel('V', fontsize = 15)
plt.grid()
plt.legend()
plt.show()

