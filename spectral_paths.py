'''
Driver code for
	--> graphs (drawing them, finding shortest paths, compute eigenvectors, etc.)
	--> spectral solutions (extract plan from eigenvector, compute its length, etc.)

Also code for generating some specialised graphs.
'''

__author__ = "Johannes Schmalz"
__year__   = 2025

import networkx as nx
import numpy as np
from numpy.linalg import eigh
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import random
import logging
import copy

# Use _default_value as an object that returns FALSE for all comparisons
#
# THIS IS A WORKAROUND
#
# using this instead of None or np.nan
# --> None is problematic because then 0 is None --> true
# --> np.nan is problematic because np.isnan(string) breaks
_default_value = object()

class Graph:
	G = None
	nodelist = []
	nodelist_no_target = []
	layout = None
	target = None
	def __init__(self, G : nx.Graph, target = _default_value):
		self.G = nx.Graph(G) # copy graph to avoid weird ownership issues
		nx.freeze(self.G) # freeze the graph to disallow modifications
		self.nodelist = list(self.G.nodes)

		# By default: we set target to the last element in nodelist
		if target == _default_value:
			target = self.nodelist[-1]
		self.set_target(target)

	def set_target(self, u): # u : Node name
		self.target = u
		self.nodelist_no_target = copy.deepcopy(self.nodelist)
		self.nodelist_no_target.remove(u)
		assert(len(self.nodelist) == len(self.nodelist_no_target) + 1)

	def name(self, u): # u : Node name
		return u

	def draw(self, label_funcs, colours = None, custom_layout = None): # label_funcs : List of label functions

		if colours is None:
			colours = {'white': list(self.G.nodes)}

		layout = None

		if custom_layout:
			layout = custom_layout

		if not layout:
			if not self.layout:
				self.layout = nx.kamada_kawai_layout(self.G)
			layout = self.layout

		node_size = max(200, 1000*len(label_funcs))
		linewidths = 2
		goal_outline_size = node_size + 800

		# Draw outline around goal
		#
		# Note: has to be before main drawing, so that this is underneath
		nx.draw_networkx_nodes(
			self.G,
			layout,
			node_shape='s',               # Square nodes
			node_size=goal_outline_size,  # Node biggity
			node_color='white',           # Background colour of node
			edgecolors='black',           # Colour of node outline
			linewidths=linewidths,        # Thickness of node outline
			nodelist=[self.target]
		)

		# Draw the graph without labels
		nx.draw(
			self.G,
			layout,
			with_labels=False,
			node_shape='s',               # Square nodes
			node_size=node_size,          # Node biggity
			node_color='white',           # Background colour of node
			edgecolors='black',           # Colour of node outline
			linewidths=linewidths,        # Thickness of node outline
			width=4,                      # Thickness of edges
		)

		# Overdraw coloured nodes
		for colour, colour_nodes in colours.items():
			nx.draw_networkx_nodes(
				self.G,
				layout,
				node_shape='s',           # Square nodes
				node_size=node_size,      # Node biggity
				node_color=colour,        # Background colour of node
				edgecolors='black',       # Colour of node outline
				linewidths=linewidths,    # Thickness of node outline
				nodelist=colour_nodes
			)

		# Helper: make string
		def stringify(x):
			return f'{x:.2f}' if isinstance(x, float) else str(x)

		# Helper: define label for a node
		def label_f(u):
			label_str = ''
			for label_func_tag, label_func in label_funcs:
				if label_str != '':
					label_str += '\n'
				label_str += label_func_tag + ' ' + stringify(label_func(u))
			return label_str

		# Draw labels
		nx.draw_networkx_labels(
			self.G,
			layout,
			labels={u: label_f(u) for u in self.nodelist},
			font_size=12
		)


		plt.show()

		# Print out as tikz
		tikzG = nx.Graph(self.G)
		def clean_label_str(x):
			return str(x).replace(' ', '-').replace('(', '').replace(')', '').replace(',', '')
		relabelling = {u : clean_label_str(u) for u in tikzG.nodes}
		relabelled_layout = {clean_label_str(u): layout[u] for u in self.G.nodes}
		tikzG = nx.relabel_nodes(tikzG, relabelling)
		print(nx.to_latex(tikzG, relabelled_layout, node_label={clean_label_str(u): label_f(u) for u in self.nodelist}))

	def node_idx(self, u) -> int: # u : Node name
		'''
		Get node index from name
		'''
		return self.nodelist.index(u)

	def node_idx_no_target(self, u) -> int: # u : Node name
		'''
		Get node index from name IF YOU DO NOT INCLUDE TARGET

		Need this for the Dirichlet Laplacian, because it has the goal struck out
		--> this idx accounts for that
		'''
		assert(u != self.target)
		return self.nodelist_no_target.index(u)

	def Adjacency(self):
		'''
		Adjacency matrix
		'''
		return nx.adjacency_matrix(self.G, nodelist=self.nodelist).toarray()

	def Laplacian(self):
		'''
		Laplacian matrix
		'''
		return nx.laplacian_matrix(self.G, nodelist=self.nodelist).toarray()

	def Dirichlet_Laplacian(self, u): # u : Node name
		'''
		Dirichlet Laplacian matrix

		i.e., Laplacian matrix with u's column and row removed
		'''
		i = self.node_idx(u)
		return np.delete(np.delete(self.Laplacian(), i, axis=0), i, axis=1)

	def shortest_path(self, u) -> int: # u : Node name
		'''
		Get the TRUE shortest path length from u to the target

		--> In other words: h*(u)
		'''
		# Implementation: just using networkx's method. This can certainly be done more cleverly.
		#
		# Note: nx paths are sequences of states, so need to subtract one to get number of edges
		return len(nx.shortest_path(self.G, source=u, target=self.target))-1


def sorted_evalues_and_evectors(L): # L : MATRIX TYPE
	# Compute evalues and evectors
	#
	# Note: assumes Hermitian matrix, i.e. L = its complex conjugate-transpose
	evalues, evectors = eigh(L)

	# Sort by evalues (from smallest to largest)
	#
	# Note: we need to transpose the evectors
	evectors = [x for _, x in sorted(zip(evalues, np.transpose(evectors)), key=lambda x: abs(x[0]))]
	evalues = sorted(evalues)

	return evalues, evectors

def smallest_nonzero_evalue_and_evector(L): # L : MATRIX TYPE
	evalues, evectors = sorted_evalues_and_evectors(L)

	smallest_nonzero_evalue = evalues[0]
	smallest_nonzero_evector = evectors[0]

	if smallest_nonzero_evalue <= 0.0:
		print("CAREFUL: smallest_nonzero_evalue = {smallest_nonzero_evalue} <= 0")
		print("--> we expect > 0...")
		print("--> this may not be an issue, but it's a symptom of numerical issues...")

	# May have to flip sign, this is normal because evector can have positive or negative solutions
	if min(smallest_nonzero_evector) < 0.0:
		smallest_nonzero_evector = [-x for x in smallest_nonzero_evector]

	# Now if there are still negative terms (after flipping) --> we are in trouble
	if min(smallest_nonzero_evector) <= 0.0:
		print(f"BAD: smallest_has non-positive term = {min(smallest_nonzero_evector)}")
		print("--> this may be **very bad**: we no longer have a descending heuristic... ")

	logging.debug(f'smallest_nonzero_evalue = {smallest_nonzero_evalue}')

	return smallest_nonzero_evalue, smallest_nonzero_evector

class SpectralSolution:
	G = None
	smallest_nonzero_evector = None
	h_scale_term = -1.0
	dist_dict = dict()
	def __init__(self, G):
		self.G = G

		_, smallest_nonzero_evector = smallest_nonzero_evalue_and_evector(
			G.Dirichlet_Laplacian(G.target))



		self.smallest_nonzero_evector = smallest_nonzero_evector

		# Compute h scale term
		#
		# The point is that we find the maximum difference between any two vertices (according to
		# their value in the smallest_nonzero_evector), and then rescale that value to be 1
		#
		# This is guaranteed to satisfy consistency constraints V(s) <= C(s,a,s') + V(s') where
		# C(s,a,s') is 1 in our case.
		max_diff = max(abs(self.value(u) - self.value(v)) for u, v in G.G.edges(data=False))
		self.h_scale_term = 1 / max_diff

		logging.debug(f'max_diff = {max_diff}')
		logging.debug(f'h_scale_term = {self.h_scale_term}')

		self.compute_dist()

	def value(self, u) -> float: # u : Node name
		'''
		Unscaled value of u in smallest eigenvector
		'''
		# Target state is undefined in evector, we say it is 0
		if u == self.G.target:
			return 0

		# Note: need to use node_idx_no_target because the Dirichlet Laplacian has the goal struck
		# out, and this idx accounts for that
		return self.smallest_nonzero_evector[self.G.node_idx_no_target(u)]

	def h(self, u) -> float: # u : Node name
		'''
		Consistent heuristic for u

		That is, the u'th term in the eigenvector after scaling
		'''
		h = self.value(u) * self.h_scale_term
		assert(h >= 0.0)
		return h

	def greedy_successor(self, u): # u : Node name, OUTPUT : Node name
		'''
		Neighbour vertex with smallest ``value''
		'''
		successors = [(v, self.value(v)) for v in self.G.G.neighbors(u)]
		greedy_succ_pair = min(successors, key=lambda x : x[1])
		return greedy_succ_pair[0]

	def compute_dist(self):
		'''
		Determine the number of steps it takes to get from any state to the target when greedily
		following ``value''

		Optimisation: compute this in topological order
		'''
		self.dist_dict = dict()

		# The target has distance 0 to itself
		self.dist_dict[self.G.target] = 0

		# Do a kind of Bellman-Ford to compute all other distances
		update_made = True
		while update_made:
			update_made = False
			for u in self.G.G.nodes:
				if u in self.dist_dict:
					continue
				if self.greedy_successor(u) in self.dist_dict:
					self.dist_dict[u] = self.dist_dict[self.greedy_successor(u)] + 1
					update_made = True

	def dist(self, u) -> int: # u : Node name
		'''
		Return the distance computed by compute_dist
		'''
		return self.dist_dict[u]

	def overshoots(self) -> list[int]:
		'''
		Returns the overshoots for all nodes

		Overshoot is how much the greedy plan (from following values greedily with
		greedy_successor) exceeds (or overshoots) the true distance dist.
		'''
		return {u: (self.dist(u) - self.G.shortest_path(u)) for u in self.G.G.nodes}


def random_connected_graph(max_nodes, seed):
    '''
    Generates a random graph that is CONNECTED. Its number of nodes will be <= max_nodes.
    '''
    random.seed(a=seed, version=2)

	# edge_prob in [0.2, 0.8]
    edge_prob = random.random() * 0.6 + 0.2
    R = nx.fast_gnp_random_graph(max_nodes, edge_prob, seed)

	# Extract largest component
    max_component_size = -1
    max_component = None
    for component in nx.connected_components(R):
        if len(component) > max_component_size:
            max_component_size = len(component)
            max_component = component
    R = R.subgraph(max_component)

    return R

def barnette_bosak_lederberg():
	'''
	Barnette-Bosák-Lederberg Graph
	https://houseofgraphs.org/graphs/954
	'''
	# Populate from left to right, then bottom to top
	left_half = nx.Graph()
	left_half.add_edges_from([
		((0,0), (3,0)), ((0,0), (0,10)), ((0,0), (1,1)),
		((0,10), (1,9)), ((0,10), (5,10)),
		((1,1), (1,5)), ((1,1), (2,2)),
		((1,5), (1,9)), ((1,5), (2,5)),
		((1,9), (2,8)),
		((2,2), (2,4)), ((2,2), (3,3)),
		((2,4), (3,4)), ((2,4), (2,5)),
		((2,5), (2,6)),
		((2,6), (3,6)), ((2,6), (2,8)),
		((2,8), (3,7)),
		((3,0), (4,0)), ((3,0), (3,3)),
		((3,3), (3,4)),
		((3,4), (3,6)),
		((3,6), (3,7)),
		((3,7), (4,8)),
		((4,0), (6,0)), ((4,0), (4,5)),
		((4,5), (6,5)), ((4,5), (4,8)),
		((4,8), (5,9)),
		((5,9), (5,10)),
		((5,10), (10,10)),
	])

	# The graph is symmetric, so extract right half from left half
	G = nx.Graph(left_half)
	for (x1, y1), (x2, y2) in left_half.edges:
		G.add_edge((10-x1, y1), (10-x2, y2))

	# Checking some properties of BBL graph
	# https://mathworld.wolfram.com/Barnette-Bosak-LederbergGraph.html
	assert(nx.is_k_regular(G, 3))
	assert(len(G.nodes) == 38)
	assert(len(G.edges) == 57)
	assert(nx.girth(G) == 4)

	return G
