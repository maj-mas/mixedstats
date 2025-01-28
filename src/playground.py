import scipy as scp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def V(x, q):
    return 4 * (1 / np.abs(x-q)**12 - 1 / np.abs(x-q)**6)

def main():
    L = 10
    m = 1
    N_C = 1
    N_Q = 1
    G = 200 # 1d grid points
    P_CUTOFF = 10 # !?

    xs = np.linspace(-L/2, L/2, G)
    ps = np.linspace(0, P_CUTOFF, G)
    dx = xs[1] - xs[0]

    states_given_x = {}
    energies_given_x = {}

    for x in tqdm(xs):
        H = np.zeros((G, G)) + np.diag(V(0, xs)) + 2/dx**2 * np.identity(G) - 1/dx**2 * np.diag((G-1) * [1.0], k=1) - 1/dx**2 * np.diag((G-1) * [1.0], k=-1)
        energies_given_x[x], states_given_x[x] = np.linalg.eigh(H)

    energies_given_xp = {}
    for p in tqdm(ps):
        for x in xs:
            energies_given_xp[(x,p)] = energies_given_x[x] + p**2/(2*m)

    betas = np.logspace(10, 0, 100)
    Z = np.zeros(100)

    for i, beta in tqdm(enumerate(betas)):
        for x in xs:
            for p in ps:
                Z[i] += np.sum(np.exp(-beta * energies_given_xp[(x,p)]))

    fig, ax = plt.subplots()
    ax.plot(betas, Z)
    ax.set_yscale("log")
    ax.set_xscale("log")
    #ax.set_ylim(1e-3, 1e8)
    fig.savefig("test.pdf")




if __name__ == "__main__":
    main()