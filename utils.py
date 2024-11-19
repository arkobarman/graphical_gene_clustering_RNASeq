import anndata
import scipy.sparse as sp
from CSCORE import CSCORE 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from node2vec import Node2Vec
import numpy as np
from sklearn.cluster import SpectralClustering
from umap import UMAP
from scipy.stats import f

def compute_top_genes(data, num_genes, gene_info):
    mean_exp = (data.values).sum(axis=0) / len(data)
    mean_exp_df = pd.DataFrame({'gene': pd.DataFrame(index=data.columns).index, 'mean_expression': mean_exp})
    top_genes_df = mean_exp_df.sort_values(by='mean_expression', ascending=False).head(num_genes)
    top_gene_indices = top_genes_df.index.astype(int).to_numpy()
    gene_names = gene_info.loc[top_gene_indices]['external_gene_name']
    
    return top_gene_indices, gene_names

def calc_CSCORE(data, num_genes, gene_info, top_gene_indices=None):
    """
    Processes input data and computes CSCORE with top genes.
    
    Parameters:
    - data: pd.DataFrame
        The input data matrix (patients * genes).
    - num_genes: int
        The number of top genes to consider based on mean expression.
    - gene_info: pd.DataFrame
        DataFrame containing gene information, must have a column 'external_gene_name'.
    
    Returns:
    - res: list
        The result of CSCORE computation, with the first element strictly symmetric.
    - gene_names: pd.Series
        Names of the top genes based on mean expression.
    """
    # Initialize obs and var
    sparse_matrix = sp.csr_matrix(data)
    obs = pd.DataFrame(index=data.index.astype(str))
    var = pd.DataFrame(index=data.columns.astype(str))

    # Create AnnData object
    adata = anndata.AnnData(X=sparse_matrix, obs=obs, var=var)
    adata.raw = adata

    # Compute mean expression and top genes
    if top_gene_indices is None:
        top_gene_indices, gene_names = compute_top_genes(data, adata, num_genes, gene_info)
    else:
        gene_names = gene_info.loc[top_gene_indices]['external_gene_name']

    # Compute CSCORE
    res = CSCORE(adata, top_gene_indices)

    # Make res[0] strictly symmetric
    for i in range(res[0].shape[0]):
        for j in range(res[0].shape[1]):
            res[0][j][i] = res[0][i][j]

    return res[0], gene_names

import seaborn as sns
import matplotlib.pyplot as plt

def plot_clustermap(matrix, gene_names, method='ward', cmap='coolwarm', figsize=(12, 12), label_fontsize=8):
    """
    Plots a hierarchical clustermap with the given matrix and gene names.

    Parameters:
    - matrix: np.ndarray or pd.DataFrame
        The input matrix for clustering and visualization (e.g., co-expression or similarity matrix).
    - gene_names: list or pd.Index
        List of gene names corresponding to the rows and columns of the matrix.
    - method: str
        Linkage method for hierarchical clustering (default is 'ward').
    - cmap: str
        Colormap for the heatmap (default is 'coolwarm').
    - figsize: tuple
        Figure size for the clustermap (default is (12, 12)).
    - label_fontsize: int
        Font size for x and y tick labels (default is 8).

    Displays:
    - A clustermap with hierarchical clustering applied.
    """
    # Create the clustermap
    g = sns.clustermap(
        matrix, method=method, cmap=cmap, figsize=figsize, xticklabels=True, yticklabels=True
    )

    # Rotate x-axis labels and adjust label sizes
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=label_fontsize)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=label_fontsize)

    # Set gene names as labels
    g.ax_heatmap.set_xticklabels(gene_names)
    g.ax_heatmap.set_yticklabels(gene_names)

    # Show the plot
    plt.show()


def calculate_coexpression_graph(co_expression_matrix, gene_names, threshold=0.7):
    """
    Constructs a co-expression graph based on a given matrix and threshold.

    Parameters:
    - co_expression_matrix: np.ndarray
        A square matrix with co-expression values.
    - gene_names: pd.Series or list
        A list or Series of gene names corresponding to the matrix indices.
    - threshold: float
        Threshold for creating edges (default is 0.7).

    Returns:
    - G: networkx.Graph
        The constructed co-expression graph.
    """
    num_genes = co_expression_matrix.shape[0]
    genes = [f"{gene_names[i]}" for i in range(num_genes)]

    # Initialize an undirected graph
    G = nx.Graph()
    G.add_nodes_from(genes)

    # Add edges based on the threshold
    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            if co_expression_matrix[i, j] >= threshold:
                G.add_edge(genes[i], genes[j], weight=co_expression_matrix[i, j])

    return G

def visualize_coexpression_graph(G):
    """
    Visualizes a co-expression graph.

    Parameters:
    - G: networkx.Graph
        The graph to visualize.

    Displays:
    - A plot of the co-expression graph with edge weights represented by color.
    """
    # Generate layout for nodes
    pos = nx.spring_layout(G)  # Layout algorithm
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]  # Extract edge weights for color mapping

    # Create a colormap for the edges
    norm = colors.Normalize(vmin=min(weights), vmax=max(weights))
    edge_colors = [cm.Blues(norm(weight)) for weight in weights]

    # Set up plot with a colorbar
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=edge_colors, width=1)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.Blues, norm=norm)
    fig.colorbar(sm, ax=ax, label="Edge Weight (Co-expression)")
    plt.title("Co-expression Graph Visualization")
    plt.show()

def graph_to_node2vec_embeddings(G, dimensions=100, walk_length=10, num_walks=200, p=1, q=1, workers=4, window=10, min_count=1, batch_words=4):
    """
    Generates Node2Vec embeddings for a given graph.

    Parameters:
    - G: networkx.Graph
        The input graph for which embeddings are to be generated.
    - dimensions: int
        Number of dimensions for the embeddings (default is 100).
    - walk_length: int
        Length of each random walk (default is 10).
    - num_walks: int
        Number of random walks per node (default is 200).
    - p: float
        Return parameter (default is 1).
    - q: float
        In-out parameter (default is 1).
    - workers: int
        Number of workers for parallel processing (default is 4).
    - window: int
        Window size for Word2Vec (default is 10).
    - min_count: int
        Minimum count of words for Word2Vec (default is 1).
    - batch_words: int
        Batch size for Word2Vec training (default is 4).

    Returns:
    - embeddings: np.ndarray
        A 2D array where each row corresponds to the embedding of a node.
        Nodes are in the same order as G.nodes().
    """
    # Initialize Node2Vec with the graph
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, 
                        num_walks=num_walks, p=p, q=q, workers=workers)
    
    # Fit the model and generate embeddings
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    
    # Get embeddings for each node in the order of nodes
    embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    
    return embeddings

def plot_dendrogram_with_embeddings(embeddings, gene_names_list, dimensions, p, q, n_clusters=5, random_state=0):
    """
    Performs spectral clustering on embeddings, reduces dimensions using UMAP, and visualizes the results.

    Parameters:
    - embeddings: np.ndarray
        The embedding matrix (nodes × dimensions).
    - gene_names_list: list
        List of gene names corresponding to the embeddings.
    - dimensions: int
        Dimensionality of the embeddings.
    - p: float
        Return parameter for Node2Vec.
    - q: float
        In-out parameter for Node2Vec.
    - n_clusters: int
        Number of clusters for spectral clustering (default is 5).
    - random_state: int
        Random state for reproducibility (default is 0).
    - output_dir: str
        Directory to save the plot (default is "figures/spectral_clustering").

    Returns:
    - labels: np.ndarray
        Cluster labels for each embedding.
    """
    # Perform spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity='rbf')
    labels = spectral.fit_predict(embeddings)

    # Reduce dimensionality with UMAP
    umap = UMAP(n_components=2, random_state=random_state)
    x_reduced = umap.fit_transform(embeddings)

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels, s=50, cmap='viridis')
    plt.title(f"Spectral Clustering Results with {n_clusters} Clusters & embedding size {dimensions} & p={p}, q={q}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    # Annotate with gene names
    for i, name in enumerate(gene_names_list):
        plt.text(x_reduced[i, 0], x_reduced[i, 1], name, fontsize=8, ha='right')

    plt.show()

    return labels

def f_test_gene_variances(matrix1, matrix2):
    """
    Perform an F-test to identify genes with significant variance differences
    between two gene expression matrices.

    Parameters:
    - matrix1: pd.DataFrame
        First gene expression matrix (samples * genes).
    - matrix2: pd.DataFrame
        Second gene expression matrix (samples * genes).

    Returns:
    - results: pd.DataFrame
        A DataFrame with columns ['gene', 'f_statistic', 'p_value'].
    """
    # Ensure the matrices have the same genes
    common_genes = matrix1.columns.intersection(matrix2.columns)
    matrix1 = matrix1[common_genes]
    matrix2 = matrix2[common_genes]

    # Calculate variances for each gene
    var1 = matrix1.var(axis=0, ddof=1)
    var2 = matrix2.var(axis=0, ddof=1)

    # Calculate F-statistic and degrees of freedom
    f_statistic = var1 / var2
    df1 = matrix1.shape[0] - 1
    df2 = matrix2.shape[0] - 1

    # Calculate p-values
    p_values = 2 * np.minimum(
        f.cdf(f_statistic, df1, df2), 1 - f.cdf(f_statistic, df1, df2)
    )

    # Combine results into a DataFrame
    results = pd.DataFrame({
        'gene': common_genes,
        'f_statistic': f_statistic,
        'p_value': p_values
    })

    return results
