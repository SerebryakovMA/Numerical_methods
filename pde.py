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
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
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
num_t = 10
u = np.array([math.exp(-10*np.power(x,2)/(num_x**2)) for x in range(-x_0*num_x, x_0*num_x, 1)])
u_prev = np.array([-dt/(2*np.power(dx,2)) for x in range(-x_0*num_x, x_0*num_x - 1, 1)])
u_i = np.array([1j + dt/(2*np.power(dx,2)) for x in range(-x_0*num_x, x_0*num_x, 1)])
u_next = u_prev

u_list = u_i.reshape(1, u_i.shape[0])

for t in range(1, num_t):
    sol1 = TDMAsolver(u_prev, u_i, u_next, u)
    #np.vectorize(sol1)
    u = (sol2(sol1, dt)).real
    u_list = np.vstack((u_list, u))

print(u_list.shape)
x = [dx*(-num_x + num) for num in range(num_x)]
t = np.array([dt*num for num in range(num_t)])
x, t = np.meshgrid(x,t)
print(x.shape)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, t, u_list)