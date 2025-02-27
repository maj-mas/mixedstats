# external imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import scipy.integrate as scpint
import scipy.optimize as scpopt
from numba import njit, prange
import pandas as pd
from multiprocessing import Pool


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
sum_cutoff = 1e4 # may be to small but we need 
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

#@njit
def I(beta, g):
    #i = lambda x: i_s(x, beta, g)
    integral, err = scpint.quad(i_s, 0, np.pi, limit=int(1e6), args=(beta, g)) # can be lowered
    return integral

@njit
def V(g, n1, n2): # yikes
    return \
        2*g/np.pi * (1/(n1-n2)*(np.cos(np.pi*(n1-n2)) - 1) - 1/(n1+n2)*(np.cos(np.pi*(n1+n2)) - 1)) \
        if n1 != n2 else \
        (2*g if n1 % 2 == 0 else g*(2 - 1/(np.pi*n1)*(np.cos(2*np.pi*n1) - 1)))
#V = np.vectorize(V)

#@njit
def Z_qc_2_L(beta, g): # Z divided by L!
    return np.sqrt(1/(4*beta)) / np.pi * th3(np.exp(-beta)) * I(beta, g)

@njit
def Z_q_1(beta):
    return th3(np.exp(-beta))

@njit
def Z_c_1_L(beta): # Z divided by L!
    return np.sqrt(1/(4*beta))

@njit
def Z_q_2(beta, g):
    z = 0.0
    for n1 in ns_sum:
        for n2 in ns_sum:
            z += np.exp(-beta*(n1**2 + n2**2 + V(g, n1, n2)))
    return z

@njit
def Z_c_2_L2(beta): # Z divided by L^2 !
    return 2/(4*beta)

@njit
def p(beta, g, L, alpha):
    ...

@njit(parallel=True)
def Zs():
    Ts = np.linspace(1e-6, 5, 200)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    Zc1_L = np.empty((len(Ts), len(gs)), dtype=np.double)
    Zc2_L2 = np.empty((len(Ts), len(gs)), dtype=np.double)
    Zq1 = np.empty((len(Ts), len(gs)), dtype=np.double)
    Zq2 = np.empty((len(Ts), len(gs)), dtype=np.double)
    
    k = 0
    for i in prange(len(Ts)):  
        print(k/100)
        k += 1
        beta = 1/Ts[i]
        zc1 = Z_c_1_L(beta)
        zc2 = Z_c_2_L2(beta)
        zq1 = Z_q_1(beta)
        for j in range(len(gs)):
            g = gs[j]
            Zc1_L[i, j] = zc1
            Zc2_L2[i, j] = zc2
            Zq1[i, j] = zq1
            Zq2[i, j] = Z_q_2(beta, g)
            #Zqc2_L[(g, beta)] = Z_qc_2_L(beta, g)

    return Zc1_L, Zc2_L2, Zq1, Zq2#, Zqc2_L

def Zqc():
    Ts = np.linspace(1e-6, 5, 200)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    Zqc2_L = np.empty((len(Ts), len(gs)), dtype=np.double)
    
    for i in range(len(Ts)):  
        print(i/100)
        beta = 1/Ts[i]
        for j in range(len(gs)):
            g = gs[j]
            Zqc2_L[i, j] = Z_qc_2_L(beta, g)

    return Zqc2_L

@njit
def fug_eqn(z, alpha, n, L, a, b, c, d, f):
    f1 = (1 - alpha)/alpha - (a*z[0] + (2*c*L - a**2*L)*z[0]**2 - (a*b + f/L)*z[0]*z[1]) / (b/L*z[1] - (2*d/L - b**2/L**2)*z[1]**2 - (a*b + f/L)*z[0]*z[1])
    f2 = n - (a*z[0] + (2*c*L - a**2*L)*z[0]**2 + b/L*z[1] + (2*d/L - b**2/L)*z[1]**2 - 2*(a*b + f)*z[0]*z[1])
    return np.array([f1, f2])

def mus(Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L):
    Ts = np.linspace(1e-6, 5, 200)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    Ls = np.logspace(-6, 2, 100)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 25, 8)

    fugs_c = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    fugs_q = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    z0 = np.array([0.01, -0.01])
    for i, T in tqdm(enumerate(Ts)):
        for j, g in enumerate(gs):
            for k, L in enumerate(Ls):
                for l, alpha in enumerate(alphas):
                    for m, n in enumerate(ns):
                        fugs_c[i, j, k, l, m], fugs_q[i, j, k, l, m] = scpopt.fsolve(fug_eqn, z0, args=(alpha, n, L, Zc1_L[i, j], Zq1[i, j], Zc2_L2[i, j], Zq2[i, j], Zqc2_L[i, j]))

    return fugs_c, fugs_q

@njit
def p_eos(beta, z, n, L, a, b, c, d, f):
    return 1/beta * (
          n/beta 
        - 0.5*(2*c - a**2)*z[0]**2 * L**2
        - 0.5*(2*d - b**2)*z[1]**2
        + (a*b + f)*z[0]*z[1] * L
    )

def save_zs():
    Zc1_L, Zc2_L2, Zq1, Zq2 = Zs()
    np.savetxt("Zc1_L.txt", Zc1_L)
    np.savetxt("Zc2_L2.txt", Zc2_L2)
    np.savetxt("Zq1.txt", Zq1)
    np.savetxt("Zq2.txt", Zq2)
    Zqc2_L = Zqc()
    np.savetxt("Zqc2_L.txt", Zqc2_L)

def load_zs():
    Zc1_L = np.loadtxt("Zc1_L.txt")
    Zc2_L2 = np.loadtxt("Zc2_L2.txt")
    Zq1 = np.loadtxt("Zq1.txt")
    Zq2 = np.loadtxt("Zq2.txt")
    Zqc2_L = np.loadtxt("Zqc2_L.txt")
    return Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L

def save_fugs():
    Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L = load_zs()
    fugs_c, fugs_q = mus(Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L)
    np.save("fugs_c.npy", fugs_c)
    np.save("fugs_q.npy", fugs_q)

def load_fugs():
    fugs_c = np.load("fugs_c.npy")
    fugs_q = np.load("fugs_q.npy")
    return fugs_c, fugs_q

def save_ps():
    Ts = np.linspace(1e-6, 5, 200)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    Ls = np.logspace(-6, 2, 100)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 25, 8)
    Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L = load_zs()
    fugs_c, fugs_q = load_fugs()

    ps = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))

    for i, T in tqdm(enumerate(Ts)):
        for j, g in enumerate(gs):
            for k, L in enumerate(Ls):
                for l, alpha in enumerate(alphas):
                    for m, n in enumerate(ns):
                        ps[i, j, k, l, m] = p_eos(1/T, 
                                                  [fugs_c[i, j, k, l, m], fugs_q[i, j, k, l, m]],
                                                  n,
                                                  L,
                                                  Zc1_L[i, j],
                                                  Zq1[i, j],
                                                  Zc2_L2[i, j],
                                                  Zq2[i, j],
                                                  Zqc2_L[i, j])
    
    np.save("ps.npy", ps)
    

def plot_mus():
    ...

def plot_ps():
    ...

#save_fugs()
#save_ps()