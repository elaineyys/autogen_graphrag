import networkx as nx
import matplotlib.pyplot as plt

# Load the graph from the .graphml file
# graph_path = "/Users/elaineyys/Desktop/autogen_graphRAG/output/20241107-151951/artifacts/merged_graph.graphml"
# graph_path = "/Users/elaineyys/Desktop/autogen_graphRAG/output/20241107-151951/artifacts/embedded_graph.graphml"
graph_path = "/Users/elaineyys/Desktop/autogen_graphRAG/output/20241126-173731/artifacts/embedded_graph.graphml"
G = nx.read_graphml(graph_path)

# # Using a spring layout for better aesthetics
# pos = nx.spring_layout(G)

# plt.figure(figsize=(12, 12))
# nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=700, font_size=12, font_color='darkblue')
# plt.title("Graph Visualization")
# plt.show()


from pyvis.network import Network

net = Network(notebook=True)
net.from_nx(G)
net.show("graph.html")