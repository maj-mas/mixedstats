# external imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from tqdm.auto import tqdm
import scipy.integrate as scpint
import scipy.optimize as scpopt
from numba import njit, prange
import pandas as pd
from multiprocessing import Pool, pool, TimeoutError
import time


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
    Ts = np.linspace(1e-6, 5, 100)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
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
    f1 = n*(1-alpha) - (a*z[0] + (2*c*L - a**2*L)*z[0]**2 - (a*b + f)*z[0]*z[1])
    f2 = alpha*n - (b/L*z[1] + (2*d/L - b**2/L)*z[1]**2 - (a*b + f)*z[0]*z[1])
    return np.array([f1, f2])
    #return f1**2 + f2**2·

finish_count = 0
needed = 100 * 5 * 50 * 6 * 6
def fugs(Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L):
    global finish_count, needed
    Ts = np.linspace(1e-6, 5, 100)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Ls = np.logspace(-6, 1, 50)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 20, 6)

    fugs_c = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    fugs_q = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    errs = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    z0 = np.array([0, 0])

    finish_count = 0
    def progress(results):
        global finish_count, needed
        if finish_count % 1000 == 0:
            print(f"{finish_count/needed}", end="\r")
        finish_count += 1

    promises = [[[[[[] for m in range(len(ns))] for l in range(len(alphas))] for k in range(len(Ls))] for j in range(len(gs))] for i in range(len(Ts))]
    with Pool(processes=20) as procs:
        for i, T in enumerate(Ts):
            for j, g in enumerate(gs):
                for k, L in enumerate(Ls):
                    for l, alpha in enumerate(alphas):
                        for m, n in enumerate(ns):
                            promises[i][j][k][l][m] = procs.apply_async(
                                scpopt.root, 
                                (fug_eqn, z0), 
                                dict(args=(alpha, n, L, Zc1_L[i, j], Zq1[i, j], Zc2_L2[i, j], Zq2[i, j], Zqc2_L[i, j])),      
                                callback=progress                          
                            )
        
        procs.close()     
        
        #time.sleep(6)      

        for i, T in enumerate(Ts):
                for j, g in enumerate(gs):
                    for k, L in enumerate(Ls):
                        for l, alpha in enumerate(alphas):
                            for m, n in enumerate(ns):
                                try:
                                    z_vec = promises[i][j][k][l][m].get().x
                                    fugs_c[i, j, k, l, m], fugs_q[i, j, k, l, m] = z_vec
                                    errs[i, j, k, l, m] = np.linalg.norm(fug_eqn(z_vec, alpha, n, L, Zc1_L[i, j], Zq1[i, j], Zc2_L2[i, j], Zq2[i, j], Zqc2_L[i, j]))
                                except TimeoutError:
                                    #print("missing sol")
                                    fugs_c[i, j, k, l, m], fugs_q[i, j, k, l, m] = [np.nan, np.nan]

        procs.join()
    
    return fugs_c, fugs_q, errs

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
    fugs_c, fugs_q, errs = fugs(Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L)
    np.save("fugs_c.npy", fugs_c)
    np.save("fugs_q.npy", fugs_q)
    np.save("errs.npy", errs)

def load_fugs():
    fugs_c = np.load("fugs_c.npy")
    fugs_q = np.load("fugs_q.npy")
    errs = np.load("errs.npy")
    return fugs_c, fugs_q, errs

def save_ps():
    Ts = np.linspace(1e-6, 5, 100)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Ls = np.logspace(-6, 1, 50)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 20, 6)
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
    Ts = np.linspace(1e-6, 5, 100)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Ls = np.logspace(-6, 1, 50)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 20, 6)
    fugs_c, fugs_q, errs = load_fugs()
    
    mus_c = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    mus_q = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    for i, T in enumerate(Ts):
        mus_c[i, ...] = np.log(fugs_c[i, ...]) / Ts[i]
        mus_q[i, ...] = np.log(fugs_q[i, ...]) / Ts[i]

    rng = np.random.default_rng()
    random_g_is = rng.integers(len(gs), high=None, size=2)
    random_alpha_is = rng.integers(len(alphas), high=None, size=2)
    random_n_is = rng.integers(len(ns), high=None, size=2)
    for j in random_g_is:
        g = gs[j]
        for l in random_alpha_is:
            alpha = alphas[l]
            for m in random_n_is:
                n = ns[m]
                level_min = min(np.nanmin(fugs_c[:, j, :, l, m]), np.nanmin(fugs_q[:, j, :, l, m]))
                level_max = max(np.nanmax(fugs_c[:, j, :, l, m]), np.nanmax(fugs_q[:, j, :, l, m]))
                #levels_exp = np.arange(np.floor(np.log10(level_min)-1), np.ceil(np.log10(level_max)+1), step=0.1)
                #levels = np.power(10, levels_exp)
                levels = np.linspace(level_min, level_max+1e-6, 100)
                fig, [c_ax, q_ax, cb1,  err_ax, cb2] = plt.subplots(figsize=(12, 6), nrows=1, ncols=5, squeeze=True, width_ratios=(1, 1, 0.3, 1, 0.3))
                cfc = c_ax.contourf(Ts, Ls, fugs_c[:, j, :, l, m].T, levels=levels)#, norm=LogNorm())
                cfq = q_ax.contourf(Ts, Ls, fugs_q[:, j, :, l, m].T, levels=levels)#, norm=LogNorm())
                fig.colorbar(cfq, cax=cb1, label="$z$", fraction=0.4)
                errfq = err_ax.contourf(Ts, Ls, np.abs(errs[:, j, :, l, m].T))
                fig.colorbar(errfq, cax=cb2, label="$f(z)\\overset{!}{=}0$", fraction=0.4)
                c_ax.set_xlabel("$T$ $(\\varepsilon)$")
                c_ax.set_yscale("log")
                q_ax.set_yscale("log")
                err_ax.set_yscale("log")
                q_ax.set_xlabel("$T$ $(\\varepsilon)$")
                c_ax.set_ylabel("$L$")
                c_ax.set_title("$z_\\text{c}$")
                q_ax.set_title("$z_\\text{q}$")
                c_ax.set_xlim(0, 5)
                q_ax.set_xlim(0, 5)
                c_ax.set_ylim(1e-6, 10)
                q_ax.set_ylim(1e-6, 10)
                err_ax.set_ylim(1e-6, 10)
                fig.suptitle(f"$g={g}\\,\\varepsilon$, $\\alpha={alpha}$, $n={np.format_float_scientific(n, precision=3)}/L$")

                fig.tight_layout()
                fig.savefig(f"../plots/int-mix/mus/fug_g{g}_alpha{alpha}_n{n}.pdf")
                plt.close(fig)
        

def plot_ps():
    ...

# save_zs()
# save_fugs()
# save_ps()
plot_mus()