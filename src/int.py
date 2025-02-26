# external imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate as scpint
from tqdm.auto import tqdm

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

def i_s(x, beta, g):
    return np.exp(-beta * g * np.sin(x)**2)


Ts = np.linspace(1e-6, 10, 1000)
gs = [0.01, 0.1, 0.5, 1, 2, 5]
betas = 1 / Ts

fig, ax = plt.subplots()
for g in gs:
    ints = []
    errs = []
    for T in tqdm(Ts):
        i = lambda x: i_s(x, 1/T, g)
        n_int, n_err = scpint.quad(i, 0, np.pi, limit=int(1e6))
        ints.append(1/np.pi * n_int)
        errs.append(1/np.pi * n_err)
    ax.plot(Ts, ints, label=f"$g={g}\\,\\varepsilon$")

ax.set_xlabel("$T$ $(\\varepsilon)$")
ax.set_ylabel("$I(\\beta)$ $(L)$")
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.grid()
ax.legend(loc="lower right")

fig.tight_layout()
fig.savefig("expsinint.pdf")