# external imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, AsinhNorm
import seaborn as sns
from tqdm.auto import tqdm
import scipy.integrate as scpint
import scipy.optimize as scpopt
from numba import njit, prange
import pandas as pd
from multiprocessing import Pool, pool, TimeoutError
import time
from mpmath import jtheta, diff, mpf, workdps


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
# pi^2 hbar ^2 / 2 m = 1
sum_cutoff = 1e4 # may be to small but we need 
ns_sum = np.arange(1, sum_cutoff, 1)
ns_sum_sq = ns_sum ** 2

Ts = np.logspace(-1, 3, 50)
gs = np.concat((np.asarray([0.0]), np.logspace(-3, 3, 7)))
Ls = np.logspace(-5, -1, 20)
alphas = np.linspace(1e-6, 1, 6)
Ns = np.logspace(10, 20, 6)

EPS_KB = 5.98e-19 # m^2 K
KB = 1.381e-23 # J / K
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
    return float(jtheta(3, 0, x)) if x <= 0.9999999 else np.sqrt(np.pi*(1-x))/(1-x) if x < 1 else np.nan
    # https://math.stackexchange.com/questions/4260536/about-the-asymptotic-behavior-of-specific-jacobi-theta-function-operatornam

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
def Z_qc_2(beta, g, L):
    t3 = th3(np.exp(-beta*EPS_KB/L**2))
    ret = np.sqrt(1/(4*beta)) / np.pi * 0.5 * (th3(np.exp(-beta*EPS_KB/L**2)) - 1) * I(beta, g) * L if not np.isnan(t3) \
        else np.sqrt(1/(4*beta)) / np.pi * Z_c_1(beta, L) * (1 - np.sqrt(2*beta*EPS_KB/np.pi)/L) * I(beta, g) * L
    return ret 

#@njit
def Z_q_1(beta, L):
    t3 = th3(np.exp(-beta*EPS_KB/L**2))
    return 0.5 * (t3 - 1) if not np.isnan(t3) else Z_c_1(beta, L) * (1 - np.sqrt(2*beta*EPS_KB/np.pi)/L)

@njit
def Z_c_1(beta, L):
    return np.sqrt(np.pi/(2*beta*EPS_KB)) * L

@njit
def Z_q_2(beta, g, L):
    # z = 0.0
    # for n1 in ns_sum:
    #     for n2 in ns_sum:
    #         z += np.exp(-beta*(n1**2*EPS_KB/L**2 + n2**2*EPS_KB/L**2 + V(g, n1, n2)))
    # return z
    return Z_c_2(beta, L) * (1 - 4*4*beta*EPS_KB/(np.sqrt(2)*L*np.pi))

@njit
def Z_c_2(beta, L):
    return np.pi/(4*beta*EPS_KB) * L**2


@njit(parallel=True)
def Zs():
    Zc1 = np.empty((len(Ts), len(gs), len(Ls)), dtype=np.double)
    Zc2 = np.empty((len(Ts), len(gs), len(Ls)), dtype=np.double)    
    Zq2 = np.empty((len(Ts), len(gs), len(Ls)), dtype=np.double)
    
    p = 0
    for i in prange(len(Ts)):  
        print(p/100)
        p += 1
        beta = 1/Ts[i]        
        for j in range(len(gs)):
            g = gs[j]
            for k in range(len(Ls)):
                L = Ls[k]
                Zc1[i, j, k] = Z_c_1(beta, L)
                Zc2[i, j, k] = Z_c_2(beta, L)
                Zq2[i, j, k] = Z_q_2(beta, g, L)

    return Zc1, Zc2, Zq2

def Zqc():
    Zq1 = np.empty((len(Ts), len(gs), len(Ls)), dtype=np.double)
    Zqc2 = np.empty((len(Ts), len(gs), len(Ls)), dtype=np.double)
    
    for i in tqdm(range(len(Ts))):  
        beta = 1/Ts[i]
        for j in range(len(gs)):
            g = gs[j]
            for k in range(len(Ls)):
                L = Ls[k]            
                try:
                    Zq1[i, j, k] = Z_q_1(beta, L)
                except:
                    Zq1[i, j, k] = np.nan
                try:
                    Zqc2[i, j, k] = Z_qc_2(beta, g, L)
                except:
                    Zqc2[i, j, k] = np.nan

    return Zq1, Zqc2

@njit
def fug_eqn_min(z, alpha, N, L, a, b, c, d, f):
    lambda_l2 = 0.001
    f1 = N*(1-alpha) - (a*z[0] + (2*c - a**2)*z[0]**2 - (a*b + f)*z[0]*z[1]) / L
    f2 = alpha*N - (b*z[1] + (2*d - b**2)*z[1]**2 - (a*b + f)*z[0]*z[1]) / L

    objective = f1**2 + f2**2 + lambda_l2 * (z[0]**2 + z[1]**2) # l2 regularisation

    grad = np.empty((2))
    grad[0] = 2 * lambda_l2 * z[0] - 2 * f1 * (a + 2*(2*c - a**2)*z[0] - (a*b + f)*z[1]) + 2 * f2 * (a*b + f)*z[1]
    grad[1] = 2 * lambda_l2 * z[1] + 2 * f1 * ((a*b + f)*z[0]) - 2 * f2 * (b + 2*(2*d - b**2)*z[1] - (a*b + f)*z[0])

    return objective, grad

@njit
def fug_eqn_min_new(z, alpha, N, L, a, b, c, d, f):
    lambda_l2 = 0.0
    k12 = (1-alpha)*N - 2
    k11 = (1-alpha)*N - 1
    k22 = alpha*N - 2
    k21 = alpha*N - 1
    f1 = z[0]**2*c*k12 + z[1]**2*d + z[0]*a*k11 + z[1]*b + z[0]*z[1]*f*k11 + (1-alpha)*N
    f2 = z[0]**2*c + z[1]**2*d*k22 + z[0]*a + z[1]*b*k21 + z[0]*z[1]*f*k21 + alpha*N

    objective = f1**2 + f2**2 + lambda_l2 * (z[0]**2 + z[1]**2) # l2 regularisation

    grad = np.empty((2))
    grad[0] = 2 * lambda_l2 * z[0] + 2 * f1 * (2*z[0]*c*k12 + a*k11 + z[0]*b + z[1]*f*k11) + 2 * f2 * (2*z[1]*d + b + z[0]*f*k11)
    grad[1] = 2 * lambda_l2 * z[1] + 2 * f1 * (2*z[0]*c + a + z[1]*f*k21) + 2 * f2 * (2*z[1]*d*k22 + b*k21 + z[0]*f*k21)

    return objective, grad

@njit
def fug_eqn_root(z, alpha, N, L, a, b, c, d, f):
    f1 = N*(1-alpha) - (a*z[0] + (2*c - a**2)*z[0]**2 - (a*b + f)*z[0]*z[1])
    f2 = alpha*N - (b*z[1] + (2*d - b**2)*z[1]**2 - (a*b + f)*z[0]*z[1])

    objective = np.array([f1, f2], dtype=np.double)

    jac = np.empty((2, 2))
    jac[0, 0] = - (a + 2*(2*c - a**2)*z[0] - (a*b + f)*z[1])
    jac[0, 1] = (a*b + f)*z[0]
    jac[1, 0] = (a*b + f)*z[1]
    jac[1, 1] = - (b + 2 * (2*d - b**2)*z[1] - (a*b + f)*z[0])

    return objective, jac

@njit
def fug_eqn_root_new(z, alpha, N, L, a, b, c, d, f):
    k12 = (1-alpha)*N - 2
    k11 = (1-alpha)*N - 1
    k22 = alpha*N - 2
    k21 = alpha*N - 1
    f1 = z[0]**2*c*k12 + z[1]**2*d + z[0]*a*k11 + z[1]*b + z[0]*z[1]*f*k11 + (1-alpha)*N
    f2 = z[0]**2*c + z[1]**2*d*k22 + z[0]*a + z[1]*b*k21 + z[0]*z[1]*f*k21 + alpha*N

    objective = np.array([f1, f2], dtype=np.double)

    jac = np.empty((2, 2))
    jac[0, 0] = 2*z[0]*c*k12 + a*k11 + z[0]*b + z[1]*f*k11
    jac[0, 1] = 2*z[1]*d + b + z[0]*f*k11
    jac[1, 0] = 2*z[0]*c + a + z[1]*f*k21
    jac[1, 1] = 2*z[1]*d*k22 + b*k21 + z[0]*f*k21

    return objective, jac

finish_count = 0
needed = 50 * 7 * 20 * 6 * 6
def fugs(Zc1, Zc2, Zq1, Zq2, Zqc2):
    global finish_count, needed

    fugs_c = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))
    fugs_q = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))
    errs = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))
    z0 = np.array([0.01, 0.01])

    finish_count = 0
    def progress(results):
        global finish_count, needed
        if finish_count % 1000 == 0:
            print(f"{finish_count/needed}", end="\r")
        finish_count += 1

    promises = [[[[[[] for m in range(len(Ns))] for l in range(len(alphas))] for k in range(len(Ls))] for j in range(len(gs))] for i in range(len(Ts))]
    with Pool(processes=8) as procs:
        for i, T in enumerate(Ts):
            for j, g in enumerate(gs):
                for k, L in enumerate(Ls):
                    for l, alpha in enumerate(alphas):
                        for m, n in enumerate(Ns):
                            promises[i][j][k][l][m] = procs.apply_async(
                                scpopt.minimize, 
                                (fug_eqn_min_new, z0), 
                                dict(jac=True, bounds=[(1e-10, None), (1e-10, None)], args=(alpha, n, L, Zc1[i, j, k], Zq1[i, j, k], Zc2[i, j, k], Zq2[i, j, k], Zqc2[i, j, k])),      
                                callback=progress                          
                            )
                            # promises[i][j][k][l][m] = procs.apply_async(
                            #     scpopt.root, 
                            #     (fug_eqn_root_new, z0), 
                            #     dict(jac=True, args=(alpha, n, L, Zc1[i, j, k], Zq1[i, j, k], Zc2[i, j, k], Zq2[i, j, k], Zqc2[i, j, k])),      
                            #     callback=progress                          
                            # )
        
        procs.close()     
        
        #time.sleep(6)      

        for i, T in enumerate(Ts):
                for j, g in enumerate(gs):
                    for k, L in enumerate(Ls):
                        for l, alpha in enumerate(alphas):
                            for m, n in enumerate(Ns):
                                try:
                                    z_vec = promises[i][j][k][l][m].get().x
                                    fugs_c[i, j, k, l, m], fugs_q[i, j, k, l, m] = z_vec
                                    sol_test, grad = fug_eqn_min_new(z_vec, alpha, n, L, Zc1[i, j, k], Zq1[i, j, k], Zc2[i, j, k], Zq2[i, j, k], Zqc2[i, j, k])
                                    # sol_test, grad = fug_eqn_root_new(z_vec, alpha, n, L, Zc1[i, j, k], Zq1[i, j, k], Zc2[i, j, k], Zq2[i, j, k], Zqc2[i, j, k])
                                    errs[i, j, k, l, m] = np.linalg.norm(sol_test) - 0 * np.linalg.norm(z_vec)**2
                                except TimeoutError:
                                    #print("missing sol")
                                    fugs_c[i, j, k, l, m], fugs_q[i, j, k, l, m] = [np.nan, np.nan]

        procs.join()
    
    return fugs_c, fugs_q, errs

@njit
def fugc(alpha, N, L, a, b, c, d, f): 
    rc1 = (1-alpha)/a
    rc2 = (2*(1-alpha) * (b*(a**2 - 2*(1-alpha)*c) -     alpha*a*f)) / (a**3 * b)
    rc3 = (-6*(-1 + alpha)*(b**3*(a**4 - a**2*(-5 + alpha)*(-1 + alpha)*c + 8*(-1 + alpha)**2*c**2) - a**4*alpha**2*b*d + \
                            a*alpha*(-6*(-1 + alpha)*b**2*c + a**2*((-3 + alpha)*b**2 + 2*alpha*d))*f + a**2*alpha*b*f**2))/(a**5*b**3)
    # rc1 = 1/a
    # #rc2 = 4*c/a**3 - 2*alpha/(1-alpha) * f/(a**2 * b) - 2*(1-2*alpha)/(1-alpha) / a
    # rc3 = 2 * (a**2 * b - 2 * b * c + 2 * alpha * b * c - alpha * a * f) / (a**3 * b)
    
    return N * rc1# + N**2 * rc2 + N**3 * rc3
    #return (1-alpha) * N * rc1 + (1-alpha) * N**2 * rc2

@njit
def fugq(alpha, N, L, a, b, c, d, f):
    rq1 = alpha/b
    rq2 = (2*alpha     * (a*(b**2 - 2*alpha    *d) - (1-alpha)*b*f)) / (a*b**3)
    rq3 = (6*alpha*(a**3*(b**4 - alpha*(4 + alpha)*b**2*d + 8*alpha**2*d**2) + 2*(-1 + alpha)**2*b**3*c*f + a**2*(-1 + alpha)*b*((2 + alpha)*b**2 - 6*alpha*d)*f - \
                   a*(-1 + alpha)*b**2*((-1 + alpha)*b**2*c + f**2)))/(a**3*b**5)
    # rq1 = 1/b
    # #rq2 = -4*d/b**3 - 2*(1-alpha)/alpha * f/(a * b**2) + 2*(1-2*alpha)/alpha / b
    # rq3 = 2 * (a * b**2 - 2 * alpha * a * d - b * f + alpha * b * f) / (a * b**3)
    
    return N * rq1# + N**2 * rq2 + N**3 * rq3
    #return alpha*N * rq1 + alpha * N**2 * rq2

@njit
def p_eos(beta, alpha, N, L, a, b, c, d, f):
    # zc = (1-alpha) * n * L * (alpha/(1-alpha)**2 / f - b * f * (f + a)/(f**2 - a*b))
    # zq = (1-alpha) * n * L * (f + a)/(f**2 - a*b)
    # zc = (1-alpha) / (a*b + f) * (b - 1/(n*L))
    # zq = alpha / (a*b + f) * (a - 1/(n*L))
    # zc = (1 - alpha) * n * L / ( a - b*a/2 + n*L * (a*b + f) * ((1 - 2*alpha)/(2*alpha) + alpha * np.sqrt( ((1 - 2*alpha)/alpha - b*a / (alpha * n * L * (a*b + f)))**2/4 - a/(alpha * n * L * (a*b + f))) ))
    # r2 = ( -1/2 * ( (1 - 2*alpha)/alpha - b*a/(alpha * n*L * (a*b + f)) ) - np.sqrt( ((1 - 2*alpha)/alpha - b*a / (alpha * n * L * (a*b + f)))**2/4 - a/(alpha * n * L * (a*b + f))))
    # r1 = 1 / (a - (a*b + f) * alpha * n * L * r2)
    # zc = (1-alpha)*n*L * r1
    # zq = alpha*n*L * r2
    zc = (a*b + (2*alpha - 1) * N * (a*b + f) - np.sqrt((a*b)**2 - 2*a*b * (a*b + f) * N + ((1 - 2*alpha) * N * (a*b + f))**2)) / (2*a * (a*b + f))
    zq = (a*b + (2*alpha - 1) * N * (a*b + f) - np.sqrt((a*b)**2 - 2*a*b * (a*b + f) * N + ((1 - 2*alpha) * N * (a*b + f))**2)) / (2*b * (a*b + f)) 
    # zc = (1 - alpha) * N * 2*b / (a*b + (2*alpha - 1) * N * (a*b + f) - np.sqrt((a*b)**2 - 2*a*b * (a*b + f) * N + ((1 - 2*alpha) * N * (a*b + f))**2))
    # zq = (a*b + (2*alpha - 1) * N * (a*b + f) + np.sqrt((a*b)**2 - 2*a*b * (a*b + f) * N + ((1 - 2*alpha) * N * (a*b + f))**2)) / (2*b * (a*b + f))
    # zc = (1-alpha)*N * 2 * b / (a*b - a*f + np.sqrt((a*b - a*f)**2 + 8*(1-alpha)*b*c*f))
    # zq = N * (a**2*(b-f) + 4*(1-alpha) * c * f - a * np.sqrt(a**2 * (b-f)**2 + 8*(1-alpha)*b*c*f)) / (4*c*f**2)
    # zc = N * ((1-alpha)*b - alpha*f) / (a*b - (f)**2)
    # zq = N * (alpha*a - (1-alpha)*f) / (a*b - (f)**2)
    # zc = (1-alpha) * N / (a)
    # zq = alpha * N / (b)
    # zc = (2*a*d + b*f + N*( (1-alpha)*(b**2 - 2*d) -     alpha*(a*b + f) )) / (2*b**2*c + 2*a**2*d - 4*c*d + 2*a*b*f + f**2)
    # zq = (2*b*c + a*f + N*(     alpha*(a**2 - 2*c) - (1-alpha)*(a*b + f) )) / (2*b**2*c + 2*a**2*d - 4*c*d + 2*a*b*f + f**2)
    # zc = fugc(alpha, N, L, a, b, c, d, f)
    # zq = fugq(alpha, N, L, a, b, c, d, f)
    

    return 1/(beta*KB*L*N) * np.log(a*zc + b*zq + c*zc**2 + d*zq**2 + f*zc*zq) #(a*zc + b*zq + 0.5*(2*c - a**2)*zc**2 + 0.5*(2*d - b**2)*zq**2 - (a*b + f)*zc*zq)

@njit
def p_virial(beta, alpha, N, L, a, b, c, d, f): # p/N
    return 1/(beta*L*EPS_KB) * (1 - virial_coeff(alpha, L, a, b, c, d, f)*N/L)

@njit 
def p_eos_z(beta, z, alpha, n, L, a, b, c, d, f):
    zc = z[0]
    zq = z[1]
    return 1/(beta*L) * (a*zc + b*zq + 0.5*(2*c - a**2)*zc**2 + 0.5*(2*d - b**2)*zq**2 - (a*b + f)*zc*zq)

@njit
def virial_coeff(alpha, L, a, b, c, d, f):
    return L * ( (1-alpha)/a**2 * ((1-alpha)*(2*c - a**2) - 4*c) + alpha/b**2 * (alpha*(2*d - b**2) - 4*d) - 1/(a*b) * (alpha*(1-alpha)*(a*b + f) + 2*(1-2*alpha)*f) )

def virial_coeffs():
    Zc1, Zc2, Zq1, Zq2, Zqc2 = load_zs()
    bs = np.empty((len(Ts), len(gs), len(Ls), len(alphas)))

    for i, T in tqdm(enumerate(Ts)):
        for j, g in enumerate(gs):
            for k, L in enumerate(Ls):
                for l, alpha in enumerate(alphas):                    
                    bs[i, j, k, l] = virial_coeff(alpha, L, Zc1[i, j, k], Zq1[i, j, k], Zc2[i, j, k], Zq2[i, j, k], Zqc2[i, j, k])

    return bs

def save_zs():
    Zc1, Zc2, Zq2 = Zs()
    np.save("Zc1.npy", Zc1)
    np.save("Zc2.npy", Zc2)    
    np.save("Zq2.npy", Zq2)
    Zq1, Zqc2 = Zqc()
    np.save("Zq1.npy", Zq1)
    np.save("Zqc2.npy", Zqc2)

def load_zs():
    Zc1 = np.load("Zc1.npy")
    Zc2 = np.load("Zc2.npy")
    Zq1 = np.load("Zq1.npy")
    Zq2 = np.load("Zq2.npy")
    Zqc2 = np.load("Zqc2.npy")
    return Zc1, Zc2, Zq1, Zq2, Zqc2

def save_fugs():
    Zc1, Zc2, Zq1, Zq2, Zqc2 = load_zs()
    fugs_c, fugs_q, errs = fugs(Zc1, Zc2, Zq1, Zq2, Zqc2)
    np.save("fugs_root_c.npy", fugs_c)
    np.save("fugs_root_q.npy", fugs_q)
    np.save("errs_root.npy", errs)

def load_fugs():
    fugs_c = np.load("fugs_root_c.npy")
    fugs_q = np.load("fugs_root_q.npy")
    errs = np.load("errs_root.npy")
    return fugs_c, fugs_q, errs

def save_ps_z():
    Zc1_L, Zc2_L2, Zq1, Zq2, Zqc2_L = load_zs()
    fugs_c, fugs_q, err = load_fugs()

    ps = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))

    for i, T in tqdm(enumerate(Ts)):
        for j, g in enumerate(gs):
            for k, L in enumerate(Ls):
                for l, alpha in enumerate(alphas):
                    for m, n in enumerate(Ns):
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
    Zc1, Zc2, Zq1, Zq2, Zqc2 = load_zs()

    ps = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))

    for i, T in tqdm(enumerate(Ts)):
        for j, g in enumerate(gs):
            for k, L in enumerate(Ls):
                for l, alpha in enumerate(alphas):
                    for m, n in enumerate(Ns):
                        try:
                            ps[i, j, k, l, m] = p_virial(1/T,
                                                        alpha,
                                                        n,
                                                        L,
                                                        Zc1[i, j, k],
                                                        Zq1[i, j, k],
                                                        Zc2[i, j, k],
                                                        Zq2[i, j, k],
                                                        Zqc2[i, j, k])
                        except:                            
                            ps[i, j, k, l, m] = np.nan
    
    np.save("ps.npy", ps)

def save_fugs_analytic():
    Zc1, Zc2, Zq1, Zq2, Zqc2 = load_zs()

    fug_c = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))
    fug_q = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))

    for i, T in tqdm(enumerate(Ts)):
        for j, g in enumerate(gs):
            for k, L in enumerate(Ls):
                for l, alpha in enumerate(alphas):
                    for m, n in enumerate(Ns):
                        try:
                            fug_c[i, j, k, l, m] = fugc(alpha,
                                                        n,
                                                        L,
                                                        Zc1[i, j, k],
                                                        Zq1[i, j, k],
                                                        Zc2[i, j, k],
                                                        Zq2[i, j, k],
                                                        Zqc2[i, j, k])                            
                        except ZeroDivisionError:                            
                            fug_c[i, j, k, l, m] = np.nan
                        
                        try:
                            fug_q[i, j, k, l, m] = fugq(alpha,
                                                        n,
                                                        L,
                                                        Zc1[i, j, k],
                                                        Zq1[i, j, k],
                                                        Zc2[i, j, k],
                                                        Zq2[i, j, k],
                                                        Zqc2[i, j, k])                            
                        except ZeroDivisionError:                            
                            fug_q[i, j, k, l, m] = np.nan
    
    np.save("fug_c_ana.npy", fug_c)
    np.save("fug_q_ana.npy", fug_q)

def load_ps_z():
    return np.load("ps_z.npy")

def load_ps():
    return np.load("ps.npy")
    

def plot_mus():
    fugs_c, fugs_q, errs = load_fugs()
    fugs_c[fugs_c <= 0] = np.nan
    fugs_q[fugs_q <= 0] = np.nan
    errs = np.abs(errs)
    errs = np.clip(errs, 1e-3, None)
    
    mus_c = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))
    mus_q = np.empty((len(Ts), len(gs), len(Ls), len(alphas), len(Ns)))
    for i, T in enumerate(Ts):
        mus_c[i, ...] = np.log(fugs_c[i, ...]) / Ts[i]
        mus_q[i, ...] = np.log(fugs_q[i, ...]) / Ts[i]

    rng = np.random.default_rng()
    random_g_is = rng.integers(len(gs), high=None, size=2)
    random_alpha_is = rng.integers(len(alphas), high=None, size=2)
    random_n_is = rng.integers(len(Ns), high=None, size=2)
    for j in random_g_is:
        g = gs[j]
        for l in random_alpha_is:
            alpha = alphas[l]
            for m in random_n_is:
                n = Ns[m]
                level_min = min(np.nanmin(fugs_c[:, j, :, l, m]), np.nanmin(fugs_q[:, j, :, l, m]))
                level_max = max(np.nanmax(fugs_c[:, j, :, l, m]), np.nanmax(fugs_q[:, j, :, l, m]))
                print(level_min, level_max)
                levels_exp = np.arange(np.floor(np.log10(level_min)-1)-1, np.ceil(np.log10(level_max)+1)+1, step=0.1)
                levels = np.power(10, levels_exp)
                level_min_err = np.nanmin(errs[:, j, :, l, m])
                level_max_err = np.nanmax(errs[:, j, :, l, m])
                levels_exp_err = np.arange(np.floor(np.log10(level_min_err)-1), np.ceil(np.log10(level_max_err)+1), step=0.1)
                levels_err = np.power(10, levels_exp_err)
                #levels = np.linspace(level_min, level_max+1e-6, 100)
                fig, [c_ax, q_ax, cb1] = plt.subplots(figsize=(12, 6), nrows=1, ncols=3, squeeze=True, width_ratios=(1, 1, 0.3)) # ,  err_ax, cb2 , 1, 0.3
                q_ax.sharey(c_ax)
                #err_ax.sharey(c_ax)
                cfc = c_ax.contourf(Ts, Ls, fugs_c[:, j, :, l, m].T, levels=levels, norm=LogNorm())
                cfq = q_ax.contourf(Ts, Ls, fugs_q[:, j, :, l, m].T, levels=levels, norm=LogNorm())
                fig.colorbar(cfq, cax=cb1, label="$z$", fraction=0.2)
                #errfq = err_ax.contourf(Ts, Ls, np.abs(errs[:, j, :, l, m].T), levels=levels_err, norm=LogNorm())
                #fig.colorbar(errfq, cax=cb2, label="$f(z)\\overset{!}{=}0$", fraction=0.2)
                c_ax.set_xlabel("$T$ $(\\varepsilon/L^2)$")
                c_ax.set_yscale("log")
                q_ax.set_yscale("log")
                #err_ax.set_yscale("log")
                q_ax.set_xlabel("$T$ $(\\varepsilon/L^2)$")
                #err_ax.set_xlabel("$T$ $(\\varepsilon)$")
                c_ax.set_ylabel("$L$")
                c_ax.set_title("$z_\\text{c}$")
                q_ax.set_title("$z_\\text{q}$")
                c_ax.set_xlim(0, 5)
                q_ax.set_xlim(0, 5)
                c_ax.set_ylim(1e-3, 10)
                q_ax.set_ylim(1e-3, 10)
                #err_ax.set_ylim(1e-3, 10)
                fig.suptitle(f"$g={g}\\,\\varepsilon/L^2$, $\\alpha={alpha}$, $N={np.format_float_scientific(n, precision=3)}$")

                # fig.tight_layout()
                fig.savefig(f"../plots/int-mix/mus/fug_g{g}_alpha{alpha}_n{n}.pdf")
                plt.close(fig)

def plot_mus_analytic():

    fugs_c = np.load("fug_c_ana.npy")
    fugs_q = np.load("fug_q_ana.npy")
    fugs_c[fugs_c <= 1e-16] = 1e-16    
    fugs_q[fugs_q <= 1e-16] = 1e-16

    rng = np.random.default_rng()
    random_g_is = [-2]#rng.integers(len(gs), high=None, size=2)
    random_alpha_is = rng.integers(len(alphas), high=None, size=2)
    random_n_is = rng.integers(len(Ns), high=None, size=2)
    for j in random_g_is:
        g = gs[j]
        for l in random_alpha_is:
            alpha = alphas[l]
            for m in random_n_is:
                try:
                    n = Ns[m]
                    level_min = min(np.nanmin(fugs_c[:, j, :, l, m]), np.nanmin(fugs_q[:, j, :, l, m]))
                    level_max = max(np.nanmax(fugs_c[:, j, :, l, m]), np.nanmax(fugs_q[:, j, :, l, m]))
                    print(level_min, level_max)
                    levels_exp = np.arange(np.floor(np.log10(level_min)-1)-1, np.ceil(np.log10(level_max)+1)+1, step=0.1)
                    levels = np.power(10, levels_exp)
                    #levels = np.linspace(level_min, level_max+1e-6, 100)
                    fig, [c_ax, q_ax, cb1] = plt.subplots(figsize=(12, 6), nrows=1, ncols=3, squeeze=True, width_ratios=(1, 1, 0.3))
                    q_ax.sharey(c_ax)
                    cfc = c_ax.contourf(Ts, Ls, fugs_c[:, j, :, l, m].T, levels=levels, norm=LogNorm())
                    cfq = q_ax.contourf(Ts, Ls, fugs_q[:, j, :, l, m].T, levels=levels, norm=LogNorm())
                    fig.colorbar(cfq, cax=cb1, label="$z$", fraction=0.2, ticks=np.power(10, np.arange(np.floor(np.log10(level_min)), np.ceil(np.log10(level_max)))))
                    c_ax.set_xlabel("$T$ $(\\varepsilon/L^2)$")
                    c_ax.set_yscale("log")
                    q_ax.set_yscale("log")
                    c_ax.set_xscale("log")
                    q_ax.set_xscale("log")
                    q_ax.set_xlabel("$T$ $(\\varepsilon/L^2)$")
                    c_ax.set_ylabel("$L$")
                    c_ax.set_title("$z_\\text{c}$")
                    q_ax.set_title("$z_\\text{q}$")
                    c_ax.set_xlim(1e-1, 1e3)
                    q_ax.set_xlim(1e-1, 1e3)
                    c_ax.set_ylim(1e-5, 1e-1)
                    q_ax.set_ylim(1e-5, 1e-1)
                    fig.suptitle(f"$g={g}\\,\\varepsilon/L^2$, $\\alpha={alpha}$, $N={np.format_float_scientific(n, precision=3)}$")

                    # fig.tight_layout()
                    fig.savefig(f"../plots/int-mix/mus/fug_ana_g{g}_alpha{alpha}_n{n}.pdf")
                    plt.close(fig)
                except:
                    pass
        

def plot_ps():
    ps = load_ps()    
    # ps = np.abs(ps) # rm TODO
    ps[np.isinf(ps)] = np.nan
    # ps[ps <= 0.0] = np.nan

    # rng = np.random.default_rng()
    # random_g_is = rng.integers(len(gs), high=None, size=2)
    # random_alpha_is = rng.integers(len(alphas), high=None, size=2)
    # random_n_is = rng.integers(len(Ns), high=None, size=2)
    random_g_is = [-2]
    random_alpha_is = [0, 1, 2, 3, 4, 5]
    random_n_is = [1]
    for j in random_g_is:
        g = gs[j]
        for l in random_alpha_is:
            alpha = alphas[l]
            for m in random_n_is:
                try:
                    n = Ns[m]
                    level_min = np.nanmin(ps[:, j, :, l, m])
                    level_max = np.nanmax(ps[:, j, :, l, m])
                    print(level_min, level_max)
                    levels_exp = np.arange(np.floor(np.log10(level_min)), np.ceil(np.log10(level_max)), step=0.1)
                    levels = np.power(10, levels_exp)
                    #levels = np.linspace(level_min, level_max, 100)

                    fig, ax = plt.subplots()
                    pcf = ax.contourf(Ts, Ls, ps[:, j, :, l, m].T, levels=levels, cmap="viridis", norm=LogNorm())
                    fig.colorbar(pcf, label="$p/N$ (?)", ticks=np.power(10, np.arange(np.floor(np.log10(level_min)), np.ceil(np.log10(level_max)))))

                    ax.set_xlabel("$T$ $(\\varepsilon/L^2)$")
                    ax.set_yscale("log")
                    ax.set_xscale("log")
                    ax.set_xlabel("$T$ $(\\varepsilon/L^2)$")
                    ax.set_ylabel("$L$")
                    #ax.set_xlim(1e-1, 1e8)
                    #ax.set_ylim(1e-3, 10)
                    fig.suptitle(f"$g={g}\\,\\varepsilon/L^2$, $\\alpha={alpha}$, $N={np.format_float_scientific(n, precision=3)}$")

                    fig.tight_layout()
                    fig.savefig(f"../plots/int-mix/ps/p_g{g}_alpha{alpha}_n{n}.pdf")
                    plt.close(fig)
                except:
                    pass

def plot_zs():
    Zc1, Zc2, Zq1, Zq2, Zqc2 = load_zs()
    zs = [Zc1, Zc2, Zq1, Zq2, Zqc2]
    labels = ["zc1", "zc2", "zq1", "zq2", "zcq2"]

    for j, g in enumerate(gs):
        for label, z in zip(labels, zs):
            z[z <= 1e-16] = 1e-16
            level_min = np.nanmin(z[:, j, :])
            level_max = np.nanmax(z[:, j, :])
            levels_exp = np.arange(np.floor(np.log10(level_min)), np.ceil(np.log10(level_max)), step=0.1)
            levels = np.power(10, levels_exp)
            #levels = np.linspace(level_min, level_max, 100)

            fig, ax = plt.subplots()
            zcf = ax.contourf(Ts, Ls, z[:, j, :].T, levels=levels, norm=LogNorm())
            fig.colorbar(zcf, label="$Z$", ticks=np.power(10, np.arange(np.floor(np.log10(level_min)), np.ceil(np.log10(level_max)))))
            ax.set_xlabel("$T$ $(\\varepsilon/L^2)$")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("$T$ $(\\varepsilon/L^2)$")
            ax.set_ylabel("$L$")
            #ax.set_xlim(1e-1, 1e8)
            #ax.set_ylim(1e-3, 10)
            fig.suptitle(f"$g={g}$")
            fig.tight_layout()
            fig.savefig("../plots/int-mix/zs/" + label + f"/g{g}.pdf")
            plt.close(fig)
        

def plot_pL():
    ps = load_ps()    
    # ps = np.abs(ps) # rm TODO
    ps[np.isinf(ps)] = np.nan
    #ps[ps <= 0.0] = np.nan

    T_is = [0, 2, 25, 49]
    j = -1 # g ind
    m = 3 # n ind

    for l, alpha in enumerate(alphas):
        g = gs[j]
        n = Ns[m]
        fig, ax = plt.subplots()
        for i in T_is:
            ax.plot(Ls, ps[i, j, :, l, m], label=f"$T={Ts[i]}\\,\\varepsilon$")

        ax.grid()
        ax.legend()
        ax.set_xlabel("$L$")
        ax.set_ylabel("$p/N$")
        ax.set_yscale("symlog")
        ax.set_xscale("log")
        #ax.set_xlim(0, 10)
        #ax.set_ylim(1e-1, 1e4)
        ax.set_title(f"$g={g}$, $\\alpha={alpha}$, $N={np.format_float_scientific(n, precision=3)}$")
        fig.savefig(f"../plots/int-mix/eos/g{g}_alpha{alpha}_n{n}.pdf")


def plot_virial_coeff():
    
    bs = virial_coeffs()

    j = 0 # g index
    #k = 5 # L index
    
    g = gs[j]
    #L = Ls[k]
    for k, L in enumerate(Ls):
        L = Ls[k]
        fig, ax = plt.subplots()
        for l, alpha in enumerate(alphas):
            ax.plot(Ts, bs[:, j, k, l], label=f"$\\alpha={alpha}$")
        ax.set_xscale("log")
        ax.set_xlim(1e-1, 1e3)
        ax.set_xlabel("$T$")
        ax.set_ylabel("$b(T)$ $(1/L^2)$")
        ax.set_title(f"$g={g}$, $L={L}$")
        ax.grid()
        ax.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(f"../plots/int-mix/coeff/g{g}_L{L}.pdf")
        plt.close(fig)


# save_zs()
# plot_zs()
# #save_fugs_analytic()
# #plot_mus_analytic()
# # # save_fugs()
# save_ps()
# # # # # save_ps_z()
# plot_ps() 
# plot_pL()
plot_virial_coeff()

# Zc1, Zc2, Zq1, Zq2, Zqc2 = load_zs()
# print(Zq1[:, 3, 10])
# print(Z_q_1(1/10, 0.01))
#print(0.5 * (th3(np.exp(-1/0.1**2)) - 1))
print(np.exp(-1/(100)*EPS_KB))
# print(th3(np.exp(-1/1e15)))
# print(th3(0.9999999999999999))
#print(th3(np.exp(-np.pi**2/np.abs(np.log(np.exp(-1/1e10))))))
#print(th3(np.exp(-1/1e10)))
# mp.dps = 100
# print(jtheta(3, 0, np.exp(-1/0.1**2)) - 1)
# print(float(jtheta(3, 0, np.exp(-1/0.1**2)) - 1))

# p = load_ps()
# print(p)
