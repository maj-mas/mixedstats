# external imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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

sum_cutoff = 1e6
ns_sum = np.arange(1, sum_cutoff, 1)
ns_sum_sq = ns_sum ** 2

def th3(x):
    return 1.0 + np.sum(x ** ns_sum_sq)

def th3_p(x):
    return np.sum(ns_sum_sq * x ** (ns_sum_sq - 1))

def th3_pp(x):
    return np.sum(ns_sum_sq * (ns_sum_sq - 1) * x ** (ns_sum_sq - 2))

def F_N(alpha, beta): # free erg per particle without constant term due to particle number
    return (- (1 - alpha) * 0.5 * (np.log(beta) + np.log(4.0 / np.pi)) + \
            alpha * np.log(th3(np.exp(-beta))) ) / beta

def C_N(alpha, beta): # specific heat per particle
    q = np.exp(-beta)
    return (1 - alpha) / 2 + alpha * beta**2 * q * ( \
        q * (th3(q)*th3_pp(q) - th3_p(q)**2)/th3(q)**2 + th3_p(q)/th3(q) )

def plot_thetas():
    xs = np.linspace(0, 1, 100)
    th    = np.vectorize(th3)((xs))
    th_p  = np.vectorize(th3_p)((xs))
    th_pp = np.vectorize(th3_pp)((xs))

    fig, ax = plt.subplots()
    ax.plot(xs, th,    label="$\\theta_3(q)$")
    ax.plot(xs, th_p,  label="$\\theta_3(q)'$")
    ax.plot(xs, th_pp, label="$\\theta_3(q)''$")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10)
    ax.set_xlabel("$q = \\text{e}^{-\\beta}$")
    ax.set_ylabel("$\\theta_3(q)$")
    ax.grid()

    fig.tight_layout()
    fig.savefig("theta.pdf")
    fig.savefig("theta.png")

def plot_F_T():
    Ts = np.linspace(1e-6, 3, 100)
    betas = 1 / Ts
    F = np.vectorize(F_N)
    F_0  = F(0.0, betas)
    F_01 = F(0.1, betas)
    F_05 = F(0.5, betas)
    F_09 = F(0.9, betas)
    F_1  = F(1.0, betas)

    fig, ax = plt.subplots()
    ax.plot(Ts, F_0, label="$\\alpha=0$")
    ax.plot(Ts, F_01, label="$\\alpha=0.1$")
    ax.plot(Ts, F_05, label="$\\alpha=0.5$")
    ax.plot(Ts, F_09, label="$\\alpha=0.9$")
    ax.plot(Ts, F_1, label="$\\alpha=1$")
    ax.set_ylim(-0.25, 2)
    ax.set_xlim(0, 3)
    ax.set_ylabel("$(F+ \\ln(\\alpha N)! + \\ln ((1-\\alpha)N)!)/N $ $(\\varepsilon)$")
    ax.set_xlabel("$T$ $(\\varepsilon/k)$")
    ax.grid()
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig("freeerg.pdf")
    fig.savefig("freeerg.png")

def plot_F_alpha():
    Ts = np.array([1e-6, 1e-2, 1e-1, 0.5, 2])
    betas = 1 / Ts
    alphas = np.linspace(0, 1, 20)
    F = np.vectorize(F_N)

    fig, ax = plt.subplots()
    for beta, T in zip(betas, Ts):
        f = F(alphas, beta)
        ax.plot(alphas, f, label=f"$T={T}$ $(\\varepsilon / k)$")
    ax.legend()
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 2)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$(F+ \\ln(\\alpha N)! + \\ln ((1-\\alpha)N)!)/N $ $(\\varepsilon)$")
    
    fig.tight_layout()
    fig.savefig("freeerg_alpha.pdf")
    fig.savefig("freeerg_alpha.png")

def plot_C():
    Ts = np.linspace(1e-6, 3, 100)
    betas = 1 / Ts
    C = np.vectorize(C_N)
    C_0  = C(0.0, betas)
    C_01 = C(0.1, betas)
    C_05 = C(0.5, betas)
    C_09 = C(0.9, betas)
    C_1  = C(1.0, betas)

    fig, ax = plt.subplots()
    ax.plot(Ts, C_0, label="$\\alpha=0$")
    ax.plot(Ts, C_01, label="$\\alpha=0.1$")
    ax.plot(Ts, C_05, label="$\\alpha=0.5$")
    ax.plot(Ts, C_09, label="$\\alpha=0.9$")
    ax.plot(Ts, C_1, label="$\\alpha=1$")
    #ax.set_ylim(-0.25, 2)
    ax.set_xlim(0, 3)
    ax.set_ylabel("$C_L/N$ $(\\varepsilon)$")
    ax.set_xlabel("$T$ $(\\varepsilon/k)$")
    ax.grid()
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig("C.pdf")
    fig.savefig("C.png")

def plot_C_alpha():
    Ts = np.array([1e-6, 1e-2, 1e-1, 0.5, 2])
    betas = 1 / Ts
    alphas = np.linspace(0, 1, 20)
    C = np.vectorize(C_N)

    fig, ax = plt.subplots()
    for beta, T in zip(betas, Ts):
        c = C(alphas, beta)
        ax.plot(alphas, c, label=f"$T={T}$ $(\\varepsilon / k)$")
    ax.legend()
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 2)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$C_L/N$ $(\\varepsilon)$")
    
    fig.tight_layout()
    fig.savefig("C_alpha.pdf")
    fig.savefig("C_alpha.png")

def main():
    plot_thetas()
    plot_F_T()
    plot_F_alpha()
    plot_C()
    plot_C_alpha()

if __name__ == "__main__":
    main()