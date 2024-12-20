# DHNSW

Efficient Approximate Nearest Neighbor Search via Data-Adaptive Parameter Adjustment in 
Hierarchical Navigable Small Graphs

Abstract—Hierarchical Navigable Small World (HNSW) graphs are a state-of-the-art solution for approximate nearest neighbor search, widely applied in areas like recommendation systems, computer vision, and natural language processing. However, the effectiveness of the HNSW algorithm is constrained by its reliance on static parameter settings, which do not account for variations in data density and dimensionality across different datasets. This paper introduces Dynamic HNSW, an adaptive method that dynamically adjusts key parameters — such as the M (number of connections per node) and ef (search depth) — based on both local data density and dimensionality of the dataset. The proposed approach improves flexibility and efficiency, allowing the graph to adapt to diverse data characteristics. Experimental results across multiple datasets demonstrate that Dynamic HNSW significantly reduces graph build time by up to 33.11% and memory usage by up to 32.44%, while maintaining comparable recall, thereby outperforming the conventional HNSW in both scalability and efficiency.

## Datasets
You will need to download corresponding datasets by yourself: MNIST, GloVe100K, SIFT1M, and GIST1M
