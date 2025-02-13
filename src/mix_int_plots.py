# external imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import scipy.integrate as scpint

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

mixint, mixint_err = scpint.quad(lambda x: np.exp(-np.cos(np.pi*x)**2), -0.5, 0.5, limit=int(1e6)) # result is this TIMES L!

def th3(x):
    return 1.0 + np.sum(x ** ns_sum_sq)

def th3_p(x):
    return np.sum(ns_sum_sq * x ** (ns_sum_sq - 1))

def th3_pp(x):
    return np.sum(ns_sum_sq * (ns_sum_sq - 1) * x ** (ns_sum_sq - 2))