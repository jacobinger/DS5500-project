import torch
import logging
from torch_geometric.data import HeteroData
torch.serialization.add_safe_globals([HeteroData])


# Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load the graph
# graph_path = "data/bindingdb_hetero_graph.pt"
# graph = torch.load(graph_path, weights_only=False)

# # Inspect the graph
# logger.info(f"Graph node types: {graph.node_types}")
# logger.info(f"Graph edge types: {graph.edge_types}")
# if ('ligand', 'binds_to', 'target') in graph.edge_types:
#     logger.info(f"Found 'ligand', 'binds_to', 'target' edges with edge_index shape: {graph['ligand', 'binds_to', 'target'].edge_index.shape}")
#     logger.info(f"Found 'ligand', 'binds_to', 'target' edge_attr shape: {graph['ligand', 'binds_to', 'target'].edge_attr.shape if hasattr(graph['ligand', 'binds_to', 'target'], 'edge_attr') else 'No edge_attr'}")
# else:
#     logger.warning("No 'ligand', 'binds_to', 'target' edge type found in graph")

# from Bio import SeqIO

# with open('data/BindingDBTargetSequences.fasta', 'r') as f:
#     for record in SeqIO.parse(f, "fasta"):
#         logger.info(f"FASTA ID: {record.id}")
#         break  # Print only the first record for now

graph = torch.load("data/bindingdb_hetero_graph.pt")
print(graph)
print("Ligand features:", graph['ligand'].x.shape, graph['ligand'].x.min(), graph['ligand'].x.max())
print("Target features:", graph['target'].x.shape, graph['target'].x.min(), graph['target'].x.max())
print("Edge attrs:", graph['ligand', 'binds_to', 'target'].edge_attr.shape, 
      graph['ligand', 'binds_to', 'target'].edge_attr.min(), 
      graph['ligand', 'binds_to', 'target'].edge_attr.max())