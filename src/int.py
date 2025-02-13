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

def i_s(x, n):
    return np.exp(-np.sin(n*np.pi*x)**2)

def i_c(x, n):
    return np.exp(-np.cos(n*np.pi*x)**2)

ns = np.arange(1, 100)
ints = []
errs = []

for n in tqdm(ns):
    if n % 2 == 0:
        i = lambda x: i_s(x, n)  
    else: 
        i = lambda x: i_c(x, n)
    n_int, n_err = scpint.quad(i, -0.5, 0.5, limit=int(1e6))
    ints.append(n_int)
    errs.append(n_err)

fig, ax = plt.subplots()
ax.plot(ns, ints)

fig.savefig("expsinint.pdf")