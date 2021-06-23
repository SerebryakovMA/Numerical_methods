import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy import stats, integrate
from scipy.optimize import curve_fit
from scipy.interpolate import pade
from scipy.interpolate import approximate_taylor_polynomial
import seaborn as sns
import pandas as pd
import math
import cmath

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

def TDMAsolver(a, b, c, d):
    #ac, bc, cc, dc = map(np.array, (a, b, c, d), dtype=np.cdouble)
    ac = np.array(a, dtype = np.cdouble)
    bc = np.array(b, dtype = np.cdouble)
    cc = np.array(c, dtype = np.cdouble)
    dc = np.array(d, dtype = np.cdouble)

    for i in range(1, len(d)): # assume a, b, c as diagonal elements, d as free terms column
        mc = ac[i-1]/bc[i-1]
        bc[i] -= mc*cc[i-1] 
        dc[i] -= mc*dc[i-1]
              
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for j in range(len(d)-2, -1, -1):
        xc[j] = (dc[j]-cc[j]*xc[j+1])/bc[j]

    return xc

def sol2(u, dt):
    sol = np.array([u[i]*cmath.exp(1j*np.power(abs(u[i]),2)*dt) for i in range(len(u))])
    return sol

dt = 0.001
x_0 = 1
num_x = 1000
dx = 2*x_0/num_x
num_t = 1000
u = np.array([math.exp(-10*np.power(x,2)/(num_x**2)) for x in range(-x_0*num_x, x_0*num_x, 1)])
u_prev = np.array([dt/(2*np.power(dx,2))*1j for x in range(-x_0*num_x, x_0*num_x, 1)])
u_i = np.array([1 - dt/(np.power(dx,2))*1j for x in range(-x_0*num_x, x_0*num_x, 1)])
u_next = u_prev

u_list = u_i.reshape(1, u_i.shape[0])

for t in range(1, num_t):
    sol1 = TDMAsolver(u_prev, u_i, u_next, u)
    #np.vectorize(sol1)
    u = (sol2(sol1, dt))
    u_list = np.vstack((u_list, u))

#u_list = np.imag(u_list)
x = [dx*num for num in range(-x_0*num_x, x_0*num_x, 1)]
t = np.array([dt*num for num in range(num_t)])
x, t = np.meshgrid(x,t)
print(x.shape)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$t$")
ax.set_zlabel(r"$u(x,t)$")

ax.plot_surface(x, t, np.real(u_list))
plt.savefig("pde_pass_real.pdf")
plt.cla()
ax.plot_surface(x, t, np.imag(u_list))
plt.savefig("pde_pass_imag.pdf")


u = np.array([[math.exp(-10*np.power(x,2)/(num_x**2)) for x in range(-x_0*num_x, x_0*num_x, 1)]])
def split_step(u, num_x, num_t, dt, dx, x_0):
    k = np.array([dx*num for num in range(-x_0*num_x, x_0*num_x, 1)])
    for t in range(num_t):
        for i in range(num_x):
            u[-1][i] *= cmath.exp(1j*np.abs(u[-1][i])*dt)

        u[-1] = np.fft.fft(u[-1])

        for i in range(num_x):
            u[-1][i] *= cmath.exp(-1j*np.power(k[i],2)*dt)

        u[-1] = np.fft.ifft(u[-1])
    u = np.vstack((u, u[-1]))

u = np.cdouble(u)
split_step(u, num_x, num_t, dt, dx, x_0)

plt.cla()
ax.plot_surface(x, t, np.real(u))
plt.savefig("pde_fft_real.pdf")
plt.cla()
ax.plot_surface(x, t, np.imag(u))
plt.savefig("pde_fft_imag.pdf")