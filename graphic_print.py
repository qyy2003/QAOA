import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
n=7
for i in range(n):
    G.add_node(i)
graph = np.load('grapy_in.npy')
for edge in graph:
    if(G.has_edge(edge[0],edge[1])):
        G.edges[edge[0],edge[1]]['weight']=G.edges[edge[0],edge[1]]['weight']+1
    else:
        G.add_edge(edge[0],edge[1], weight=1)
# print(list(G.nodes))
# print(list(G.edges))
pos = nx.spring_layout(G)
nx.draw_networkx(G,pos, with_labels=True, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
print(labels)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.savefig('graphic_n=7.png')
# print(graph)