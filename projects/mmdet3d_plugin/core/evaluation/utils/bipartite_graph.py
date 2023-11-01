import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

class Bipartite_graph :
    def __init__(self):
        self.G = nx.Graph()

    def add_node(self, left, right):
        self.G.add_nodes_from(left, bipartite=0)
        self.G.add_nodes_from(right, bipartite=1)

    def add_weighted_edges(self, edges):
        self.G.add_weighted_edges_from(edges)

    def match(self) :
        return dict(nx.algorithms.matching.max_weight_matching(self.G, maxcardinality=False, weight='weight' ))
        
    def draw(self):
        pos = nx.spring_layout(self.G, k=10)
        nx.draw(self.G, with_labels = True)
        labels = {e: self.G.edges[e]['weight'] for e in self.G.edges}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
        plt.show()

if __name__ == '__main__':
    G = Bipartite_graph()
    G.add_weighted_edges([('1', 'a', 2), ('2', 'b', 4), ('1', 'b', 1)])
    #G.draw()
    #print(G.match())
