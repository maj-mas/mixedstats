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
from mpmath import jtheta, diff


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

# @njit
# def th3(x):
#     return 1.0 + 2 * np.sum(x ** ns_sum_sq)

# @njit
# def th3_p(x):
#     return 2 * np.sum(ns_sum_sq * x ** (ns_sum_sq - 1))

# @njit
# def th3_pp(x):
#     return 2* np.sum(ns_sum_sq * (ns_sum_sq - 1) * x ** (ns_sum_sq - 2))

def th3(x):
    return float(jtheta(3, 0, x))

def th3_p(x):
    return float(diff(lambda x: jtheta(3, 0, x), x))

def th3_pp(x):
    return float(diff(lambda x: jtheta(3, 0, x), x, 2))

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
    return np.sqrt(1/(4*beta)) / np.pi * 0.5 * (th3(np.exp(-beta)) - 1) * I(beta, g)

#@njit
def Z_q_1(beta):
    return 0.5 * (th3(np.exp(-beta)) - 1)

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


@njit(parallel=True)
def Zs():
    Ts = np.linspace(5e-2, 5, 50)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Zc1_L = np.empty((len(Ts), len(gs)), dtype=np.double)
    Zc2_L2 = np.empty((len(Ts), len(gs)), dtype=np.double)    
    Zq2 = np.empty((len(Ts), len(gs)), dtype=np.double)
    
    k = 0
    for i in prange(len(Ts)):  
        print(k/100)
        k += 1
        beta = 1/Ts[i]
        zc1 = Z_c_1_L(beta)
        zc2 = Z_c_2_L2(beta)
        for j in range(len(gs)):
            g = gs[j]
            Zc1_L[i, j] = zc1
            Zc2_L2[i, j] = zc2
            Zq2[i, j] = Z_q_2(beta, g)
            #Zqc2_L[(g, beta)] = Z_qc_2_L(beta, g)

    return Zc1_L, Zc2_L2, Zq2#, Zqc2_L

def Zqc():
    Ts = np.linspace(5e-2, 5, 50)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    Zq1 = np.empty((len(Ts), len(gs)), dtype=np.double)
    Zqc2_L = np.empty((len(Ts), len(gs)), dtype=np.double)
    
    for i in range(len(Ts)):  
        print(i/100)
        beta = 1/Ts[i]
        for j in range(len(gs)):
            g = gs[j]
            Zq1[i, j] = Z_q_1(beta)
            Zqc2_L[i, j] = Z_qc_2_L(beta, g)

    return Zq1, Zqc2_L

@njit
def fug_eqn_min(z, alpha, n, L, a, b, c, d, f):
    lambda_l2 = 0.001
    f1 = n*(1-alpha) - (a*z[0] + (2*c*L - a**2*L)*z[0]**2 - (a*b + f)*z[0]*z[1])
    f2 = alpha*n - (b/L*z[1] + (2*d/L - b**2/L)*z[1]**2 - (a*b + f)*z[0]*z[1])

    objective = f1**2 + f2**2 + lambda_l2 * (z[0]**2 + z[1]**2) # l2 regularisation

    grad = np.empty((2))
    grad[0] = 2 * lambda_l2 * z[0] - 2 * f1 * (a + 2*(2*c*L - a**2*L)*z[0] - (a*b + f)*z[1]) + 2 * f2 * (a*b + f)*z[1]
    grad[1] = 2 * lambda_l2 * z[1] + 2 * f1 * ((a*b + f)*z[0]) - 2 * f2 * (b/L + 2*(2*d/L - b**2/L)*z[1] - (a*b + f)*z[0])

    return objective, grad

@njit
def fug_eqn_root(z, alpha, n, L, a, b, c, d, f):
    f1 = n*(1-alpha) - (a*z[0] + (2*c*L - a**2*L)*z[0]**2 - (a*b + f)*z[0]*z[1])
    f2 = alpha*n - (b/L*z[1] + (2*d/L - b**2/L)*z[1]**2 - (a*b + f)*z[0]*z[1])

    objective = np.array([f1, f2])

    jac = np.empty((2, 2))
    jac[0, 0] = - (a + 2*(2*c*L - a**2*L)*z[0] - (a*b + f)*z[1])
    jac[0, 1] = (a*b + f)*z[0]
    jac[1, 0] = (a*b + f)*z[1]
    jac[1, 1] = - (b/L + 2 * (2*d/L - b**2/L)*z[1] - (a*b + f)*z[0])

    return objective, jac

finish_count = 0
needed = 50 * 5 * 20 * 6 * 6
def fugs(Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L):
    global finish_count, needed
    Ts = np.linspace(5e-2, 5, 50)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Ls = np.logspace(-3, 1, 20)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 20, 6)

    fugs_c = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    fugs_q = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    errs = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))
    z0 = np.array([0.01, 0.01])

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
                            # promises[i][j][k][l][m] = procs.apply_async(
                            #     scpopt.minimize, 
                            #     (fug_eqn_min, z0), 
                            #     dict(jac=True, bounds=[(1e-10, None), (1e-10, None)], args=(alpha, n, L, Zc1_L[i, j], Zq1[i, j], Zc2_L2[i, j], Zq2[i, j], Zqc2_L[i, j])),      
                            #     callback=progress                          
                            # )
                            promises[i][j][k][l][m] = procs.apply_async(
                                scpopt.root, 
                                (fug_eqn_root, z0), 
                                dict(jac=True, args=(alpha, n, L, Zc1_L[i, j], Zq1[i, j], Zc2_L2[i, j], Zq2[i, j], Zqc2_L[i, j])),      
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
                                    #sol_test, grad = fug_eqn_min(z_vec, alpha, n, L, Zc1_L[i, j], Zq1[i, j], Zc2_L2[i, j], Zq2[i, j], Zqc2_L[i, j])
                                    sol_test, grad = fug_eqn_root(z_vec, alpha, n, L, Zc1_L[i, j], Zq1[i, j], Zc2_L2[i, j], Zq2[i, j], Zqc2_L[i, j])
                                    errs[i, j, k, l, m] = np.linalg.norm(sol_test)# - 0.001 * np.linalg.norm(z_vec)**2
                                except TimeoutError:
                                    #print("missing sol")
                                    fugs_c[i, j, k, l, m], fugs_q[i, j, k, l, m] = [np.nan, np.nan]

        procs.join()
    
    return fugs_c, fugs_q, errs

@njit
def p_eos(beta, alpha, n, L, a, b, c, d, f):
    zc = (1-alpha) * n * L * (alpha/(1-alpha)**2 / f - b * f * (f + a)/(f**2 - a*b))
    zq = (1-alpha) * n * L * (f + a)/(f**2 - a*b)
    return 1/(beta) * (a*zc + b/L*zq + 0.5*(2*c*L - a*L**2)*zc**2 + 0.5*(2*d/L - b**2/L)*zq**2 - (a*b + f)*zc*zq) # factors 1/L distributed

@njit 
def p_eos_z(beta, z, alpha, n, L, a, b, c, d, f):
    zc = z[0]
    zq = z[1]
    return 1/(beta) * (a*zc + b/L*zq + 0.5*(2*c*L - a*L**2)*zc**2 + 0.5*(2*d/L - b**2/L)*zq**2 - (a*b + f)*zc*zq) # factors 1/L distributed

def save_zs():
    Zc1_L, Zc2_L2, Zq2 = Zs()
    np.savetxt("Zc1_L.txt", Zc1_L)
    np.savetxt("Zc2_L2.txt", Zc2_L2)    
    np.savetxt("Zq2.txt", Zq2)
    Zq1, Zqc2_L = Zqc()
    np.savetxt("Zq1.txt", Zq1)
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
    np.save("fugs_root_c.npy", fugs_c)
    np.save("fugs_root_q.npy", fugs_q)
    np.save("errs_root.npy", errs)

def load_fugs():
    fugs_c = np.load("fugs_root_c.npy")
    fugs_q = np.load("fugs_root_q.npy")
    errs = np.load("errs_root.npy")
    return fugs_c, fugs_q, errs

def save_ps_z():
    Ts = np.linspace(5e-2, 5, 50)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Ls = np.logspace(-3, 1, 20)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 20, 6)
    Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L = load_zs()
    fugs_c, fugs_q, err = load_fugs()

    ps = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))

    for i, T in tqdm(enumerate(Ts)):
        for j, g in enumerate(gs):
            for k, L in enumerate(Ls):
                for l, alpha in enumerate(alphas):
                    for m, n in enumerate(ns):
                        try:
                            zc = fugs_c[i, j, k, l, m]
                            zq = fugs_q[i, j, k, l, m]
                            ps[i, j, k, l, m] = p_eos_z(1/T, 
                                                        [zc, zq],
                                                        alpha,
                                                        n,
                                                        L,
                                                        Zc1_L[i, j],
                                                        Zq1[i, j],
                                                        Zc2_L2[i, j],
                                                        Zq2[i, j],
                                                        Zqc2_L[i, j])
                        except:                            
                            ps[i, j, k, l, m] = np.nan
    
    np.save("ps_z.npy", ps)

def save_ps():
    Ts = np.linspace(5e-2, 5, 50)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Ls = np.logspace(-3, 1, 20)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 20, 6)
    Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L = load_zs()

    ps = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(ns)))

    for i, T in tqdm(enumerate(Ts)):
        for j, g in enumerate(gs):
            for k, L in enumerate(Ls):
                for l, alpha in enumerate(alphas):
                    for m, n in enumerate(ns):
                        try:
                            ps[i, j, k, l, m] = p_eos(1/T,
                                                        alpha,
                                                        n,
                                                        L,
                                                        Zc1_L[i, j],
                                                        Zq1[i, j],
                                                        Zc2_L2[i, j],
                                                        Zq2[i, j],
                                                        Zqc2_L[i, j])
                        except:                            
                            ps[i, j, k, l, m] = np.nan
    
    np.save("ps.npy", ps)

def load_ps_z():
    return np.load("ps_z.npy")

def load_ps():
    return np.load("ps.npy")
    

def plot_mus():
    Ts = np.linspace(5e-2, 5, 50)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Ls = np.logspace(-3, 1, 20)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 20, 6)
    fugs_c, fugs_q, errs = load_fugs()
    errs = np.abs(errs)
    errs = np.clip(errs, 1e-3, None)
    
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
                #levels_exp = np.arange(np.floor(np.log10(level_min)-1)-1, np.ceil(np.log10(level_max)+1)+1, step=0.1)
                #levels = np.power(10, levels_exp)
                level_min_err = np.nanmin(errs[:, j, :, l, m])
                level_max_err = np.nanmax(errs[:, j, :, l, m])
                levels_exp_err = np.arange(np.floor(np.log10(level_min_err)-1), np.ceil(np.log10(level_max_err)+1), step=0.1)
                levels_err = np.power(10, levels_exp_err)
                levels = np.linspace(level_min, level_max+1e-6, 100)
                fig, [c_ax, q_ax, cb1,  err_ax, cb2] = plt.subplots(figsize=(12, 6), nrows=1, ncols=5, squeeze=True, width_ratios=(1, 1, 0.3, 1, 0.3))
                cfc = c_ax.contourf(Ts, Ls, fugs_c[:, j, :, l, m].T, levels=levels)#, norm=LogNorm())
                cfq = q_ax.contourf(Ts, Ls, fugs_q[:, j, :, l, m].T, levels=levels)#, norm=LogNorm())
                fig.colorbar(cfq, cax=cb1, label="$z$", fraction=0.4)
                errfq = err_ax.contourf(Ts, Ls, np.abs(errs[:, j, :, l, m].T), levels=levels_err, norm=LogNorm())
                fig.colorbar(errfq, cax=cb2, label="$f(z)\\overset{!}{=}0$", fraction=0.4)
                c_ax.set_xlabel("$T$ $(\\varepsilon)$")
                c_ax.set_yscale("log")
                q_ax.set_yscale("log")
                err_ax.set_yscale("log")
                q_ax.set_xlabel("$T$ $(\\varepsilon)$")
                err_ax.set_xlabel("$T$ $(\\varepsilon)$")
                c_ax.set_ylabel("$L$")
                c_ax.set_title("$z_\\text{c}$")
                q_ax.set_title("$z_\\text{q}$")
                c_ax.set_xlim(0, 5)
                q_ax.set_xlim(0, 5)
                c_ax.set_ylim(1e-3, 10)
                q_ax.set_ylim(1e-3, 10)
                err_ax.set_ylim(1e-3, 10)
                fig.suptitle(f"$g={g}\\,\\varepsilon$, $\\alpha={alpha}$, $n={np.format_float_scientific(n, precision=3)}/L$")

                #fig.tight_layout()
                fig.savefig(f"../plots/int-mix/mus/fug_g{g}_alpha{alpha}_n{n}.pdf")
                plt.close(fig)
        

def plot_ps():
    Ts = np.linspace(5e-2, 5, 50)
    gs = [0.01, 0.1, 0.5, 1.0, 2.0]
    Ls = np.logspace(-3, 1, 20)
    alphas = np.linspace(1e-6, 1, 6)
    ns = np.logspace(10, 20, 6)
    ps = load_ps_z()    
    ps = np.clip(ps, 1e-6, None)

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
                level_min = np.nanmin(ps[:, j, :, l, m])
                level_max = np.nanmax(ps[:, j, :, l, m])
                levels_exp = np.arange(np.floor(np.log10(level_min)), np.ceil(np.log10(level_max)), step=0.1)
                levels = np.power(10, levels_exp)
                #levels = np.linspace(level_min, level_max, 100)

                fig, ax = plt.subplots()
                pcf = ax.contourf(Ts, Ls, ps[:, j, :, l, m].T, levels=levels, norm=LogNorm())
                fig.colorbar(pcf, label="$p$ (?)", ticks=np.power(10, np.arange(np.floor(np.log10(level_min)), np.ceil(np.log10(level_max)))))

                ax.set_xlabel("$T$ $(\\varepsilon)$")
                ax.set_yscale("log")
                ax.set_xlabel("$T$ $(\\varepsilon)$")
                ax.set_ylabel("$L$")
                ax.set_xlim(0, 5)
                ax.set_ylim(1e-3, 10)
                fig.suptitle(f"$g={g}\\,\\varepsilon$, $\\alpha={alpha}$, $n={np.format_float_scientific(n, precision=3)}/L$")

                fig.tight_layout()
                fig.savefig(f"../plots/int-mix/ps/p_g{g}_alpha{alpha}_n{n}.pdf")
                plt.close(fig)


# save_zs()
# save_fugs()
# save_ps()
# save_ps_z()
plot_ps()
# plot_mus()