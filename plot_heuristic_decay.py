'''
Generate a path graph with graph_size, compute its eigenvector, and plot their error
'''

import networkx as nx
import matplotlib.pyplot as plt
import spectral_paths as sp
import argparse
import numpy as np
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description="plot decay of spectral heuristic on path graph")
parser.add_argument("graph_type", choices=["path", "grid"], help="type of graph")
parser.add_argument("graph_size", help="size of graph (# vertices for path graph; width of grid graph)")

args = parser.parse_args()

# LaTeX support
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "pgf.rcfonts": False
})

###
### Generate graphs and data
###

if args.graph_type == "grid":
    nxG = nx.grid_2d_graph(int(args.graph_size), int(args.graph_size))
    GRID = sp.Graph(nxG, (0, 0))
    soln = sp.SpectralSolution(GRID)

    x = list(range(int(args.graph_size)))
    top_row = [(0, t) for t in range(int(args.graph_size))]
    y = [GRID.shortest_path(u) - soln.h(u) for u in top_row]

elif args.graph_type == "path":
    nxG = nx.line_graph(int(args.graph_size))
    layoutPATH = {u: (u, 0) for u in nxG.nodes}
    PATH = sp.Graph(nxG, 0)
    soln = sp.SpectralSolution(PATH)

    x = list(nxG.nodes)
    y = [PATH.shortest_path(u) - soln.h(u) for u in x]

###
### Plot data
###

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Plot data
ax.plot(x, y, color='blue', linewidth=2)

# # Add diagonal line
# ax.plot(x, x, linestyle='--', color='red', linewidth=1.5)

# Label x and y axis
ax.set_xlabel(r"Distance from Goal")
ax.set_ylabel(r"Error")

# Add a grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Print to pdf
fig.tight_layout()
fig.savefig(f"decay_plot_{args.graph_type}_{args.graph_size}.pdf")

###
### Fit best power curve
###

def exponent(x, a, b):
    return a * x**b

# Convert data to numpy arrays
x = np.asarray(x, dtype=float)
y = np.asarray(y, dtype=float)

# Avoid zero values for x
mask = x > 0
x_fit = x[mask]
y_fit = y[mask]

# Initial guess for a and b
p0 = [0.001, 2.0]

# Fit the curve
params, covariance = curve_fit(exponent, x_fit, y_fit, p0=p0, maxfev=20000)
a, b = params

print(f"Power-law fit: y = {a:.10f} * x^{b:.5f}")

# Compute R2
y_pred = exponent(x_fit, a, b)
ss_res = np.sum((y_fit - y_pred)**2)
ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
r2 = 1 - ss_res / ss_tot
print(f"R² = {r2:.5f}")
