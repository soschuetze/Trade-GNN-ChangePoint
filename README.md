# Change-Point Detection of International Trade Crises using GraphML Methods

1. Determine if change-points in the international trade network can be detected that correlate to known crises (Brexit, 1973 Oil Crisis, etc.)
2. Build s-GNN to learn graph similarity function that classifies pairs of networks as being separated by change-point or not
3. Training uses known events to determine if pair of graphs includes change-point
4. s-GNN - input is pair of feature embeddings derived from GCN encodings and output is the similarity score
5. Can analyze how changes differ by region and by product - were specific products/regions impacted by certain events or not
