'''
Script for generating the plots in the workshop paper and talks
'''

import spectral_paths as sp
import networkx as nx
import random

demos = dict()

###
### Define graphs to draw
###

### Path
nxG = nx.path_graph(12)
layoutPATH = {u : (u, 0) for u in nxG.nodes}
PATH = sp.Graph(nxG)

### Grid
nxG = nx.grid_2d_graph(5, 5)
layoutGRID = {u: u for u in nxG.nodes}
GRID = sp.Graph(nxG)

### Grid-based Maze
nxG = nx.grid_2d_graph(5, 5)
tree = nx.random_spanning_tree(nxG, seed=42)
maze = nx.Graph(nxG)
random.seed(a=42, version=2)
for (u, v) in nxG.edges:
	# Not allowed to delete edges in the tree
	if (u, v) in tree.edges:
		continue

	# Probabilistically delete edges not in tree
	if random.random() > 0.2:
		maze.remove_edge(u, v)

nxG = maze
layoutMAZE = {u: u for u in nxG.nodes}
MAZE = sp.Graph(nxG)

### Barnette-Bosak-Lederberg
nxG = sp.barnette_bosak_lederberg()
layoutBBL = {u: u for u in nxG.nodes}
BBL = sp.Graph(nxG, target=(10,0))

### House
nxG = nx.house_x_graph()
nxG.remove_edge(0,3)
HOUSE = sp.Graph(nxG)


### Tadpole
tadpole_sizes = [
	(5, (2, 5)),
	(8, (5, 8)),
	(12, (9, 12)),
	(20, (17, 20)),
	(30, (27, 30)),
	(40, (37, 40)),
]
tadpoles = dict()
for name, (path_length, cycle_length) in tadpole_sizes:
	nxG = nx.tadpole_graph(cycle_length, path_length)
	tadpoles[name] = sp.Graph(nxG, target=path_length+1)


###
### Draw the graphs
###

to_draw = [
	(HOUSE, None),
	(PATH, layoutPATH),
	(GRID, layoutGRID),
	(MAZE, layoutMAZE),
	(BBL, layoutBBL),
	(tadpoles[5], None),
	(tadpoles[8], None),
	(tadpoles[12], None),
]
for G, layout in to_draw:
	soln = sp.SpectralSolution(G)

	### Create colour gradient based on the spectral solution's overshoot
	colours = [
		"#ffffff",  # white
		"#ff9999",  # light soft red
		"#ff6666",  # medium light red
		"#ff4d4d",  # moderate red
		"#ff1a1a",  # strong red
		"#cc0000",  # deep bright red
		"#990000",  # dark red
	]
	colour_map = dict()
	for u in G.G.nodes:
		colour_idx = min(soln.dist(u) - G.shortest_path(u), len(colours)-1)
		colour = colours[colour_idx]
		if colour not in colour_map:
			colour_map[colour] = []
		colour_map[colour].append(u)

	def rounded_h(x):
		return f'{soln.h(x):.2f} '

	### Draw the thing
	G.draw([
		# ('id', G.name),
		# ('got', soln.dist),
		# ('opt', G.shortest_path),
		# ('h', soln.h),
		# ('v', soln.value),
		# ('succ', soln.greedy_successor),
		# ('diff', lambda x : G.shortest_path(x) - soln.h(x)),
		('', rounded_h),
	],
	colours=colour_map,
	custom_layout=layout)
