# external imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import scipy.integrate as scpint
from numba import njit

# mpl defaults
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{parskip}\usepackage{braket}",
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 24,
    "figure.titlesize": 24,
    "font.family": "serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})
sns.color_palette("colorblind")

# units:
# k_B = 1
# L = 1
# pi^2 hbar ^2 / 2 m = 1
sum_cutoff = 1e4 # may bee to small but we need 
ns_sum = np.arange(1, sum_cutoff, 1)
ns_sum_sq = ns_sum ** 2

@njit
def th3(x):
    return 1.0 + np.sum(x ** ns_sum_sq)

@njit
def th3_p(x):
    return np.sum(ns_sum_sq * x ** (ns_sum_sq - 1))

@njit
def th3_pp(x):
    return np.sum(ns_sum_sq * (ns_sum_sq - 1) * x ** (ns_sum_sq - 2))

@njit
def i_s(x, beta, g):
    return np.exp(-beta * g * np.sin(x)**2)

@njit
def I(beta, g):
    i = lambda x: i_s(x, beta, g)
    int, err = scpint.quad(i, 0, np.pi, limit=int(1e6)) # can be lowered
    return int

@njit
def V(g, n1, n2): # yikes
    return \
        2*g/np.pi * (1/(n1-n2)*(np.cos(np.pi*(n1-n2)) - 1) - 1/(n1+n2)*(np.cos(np.pi*(n1+n2)) - 1)) \
        if n1 != n2 else \
        (2*g if n1 % 2 == 0 else g*(2 - 1/(np.pi*n1)*(np.cos(2*np.pi*n1) - 1)))
#V = np.vectorize(V)

@njit
def Z_qc_2(beta, g, L):
    return np.sqrt(1/(4*beta)) / np.pi * th3(np.exp(-beta)) * I(beta, g) * L

@njit
def Z_q_1(beta):
    return th3(np.exp(-beta))

@njit
def Z_c_1(beta, L):
    return np.sqrt(1/(4*beta)) * L

@njit
def Z_q_2(beta, g):
    z = 0.0
    for n1 in ns_sum:
        for n2 in ns_sum:
            z += np.exp(-beta*(n1**2 + n2**2 + V(g, n1, n2)))
    return z

@njit
def Z_c_2(beta, L):
    return 2/(4*beta) * L**2

@njit
def p(beta, g, L, alpha):
    ...

print(Z_q_2(0.1, 3))