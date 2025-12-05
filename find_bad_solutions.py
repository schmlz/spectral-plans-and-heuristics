'''
Simple script that searches for graphs with large spectral overshoots
'''

import spectral_paths as sp
import networkx as nx

n_vertices = [10, 15, 20]
n_seeds = 1000

print('running...')

# Iterate through the different numbers of vertices
for n in n_vertices:
	global_max_overshoot_n = 0
	# Iterate through seeds -- note: we're just going through them linearly, that's good enough for
	# generating our graphs.
	for seed in range(n_seeds):
		G = sp.Graph(sp.random_connected_graph(n, seed))

		# Remove graphs where, after removing the target, it becomes disconnected
		Gt = nx.Graph(G.G)
		Gt.remove_node(G.target)
		if not nx.is_connected(Gt):
			continue

		soln = sp.SpectralSolution(G)

		# Print out big overshoots
		max_overshoot = max(soln.overshoots().values())
		if max_overshoot > global_max_overshoot_n:
			print(f'--- n={n}  seed={seed} ---')
			print(f'max_overshoot = {max_overshoot}')
			global_max_overshoot_n = max_overshoot

			for key, val in soln.overshoots().items():
				if val == max_overshoot:
					print(key)
