import pandas as pd
import numpy as np
import os
from CSCORE import CSCORE 
import anndata
import scipy.sparse as sp
import random
import warnings
import pickle

import networkx as nx
from pecanpy.experimental import Node2vecPlusPlus
from gensim.models import Word2Vec
from sklearn.cluster import SpectralClustering
import pandas as pd 
import plotly.express as px
from permetrics import ClusteringMetric
import random
from scipy.sparse import csgraph
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph 
import warnings
import pickle
import statistics as s
import gc
from diptest import diptest
import optuna
import json
from functools import partial

from scipy.sparse import csgraph
from numpy import linalg as LA

import statistics as s 
import umap
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns 

def preprocess_data(kept_mgs_levels=[1, 4]):
    repo_dir = os.path.dirname(os.getcwd())
    raw_dataset_dir = os.path.join(repo_dir, 'dataset', 'raw_dataset')

    #load the the dataset with the genes and expression values 
    gene_expression = pd.read_csv(os.path.join(raw_dataset_dir, 'Genelevel_expectedcounts_matrix.tsv'), sep='\t')
    #load the data that maps the EN gene IDs to the gene names
    gene_info = pd.read_csv(os.path.join(raw_dataset_dir, 'gene_info.tsv'), sep='\t')[['ensembl_gene_id', 'external_gene_name']]
    #loading the metadata that shows the sample IDs and the mgs labels 
    meta_data = pd.read_csv(os.path.join(raw_dataset_dir, 'MetaSheet.csv'), encoding='unicode_escape')
    meta_data = meta_data[['r_id', 'mgs_level']]

    #adding joining on the on EN ids to add the gene names to the expression data 
    gene_expression = gene_expression.merge(gene_info, left_on='Unnamed: 0', right_on='ensembl_gene_id')

    #transposting to make the rows the samples and the columns the genes 
    gene_expression = gene_expression.T
    #grabbing the last row or the dataframe which contains the genes names 
    gene_namnes = gene_expression.iloc[-1]
    #drops both the EN rows and the gene names row 
    gene_expression.drop([gene_expression.index[0], gene_expression.index[-2]], inplace=True)
    #renaming the columns to be the gene names
    gene_expression.columns = gene_expression.iloc[-1].tolist()
    #droping the last row which contains the gene names
    gene_expression.drop(gene_expression.index[-1], inplace=True)

    #add the mgs labes to the gene expression data based on the samples ids (which is the in of the rows)
    gene_expression = gene_expression.merge(meta_data, left_on=gene_expression.index, right_on='r_id')

    #mgs level to keep 
    mgs_levels = kept_mgs_levels
    gene_expression = gene_expression[gene_expression['mgs_level'].isin(mgs_levels)]
    #dropping the r_id and mgs_levels columns
    gene_expression.drop(columns=['r_id', 'mgs_level'], inplace=True)
    
    #we now have the dataframe of just samples by genes, we need to filter out genes that do not have any expression 
    vars = np.var(gene_expression.values, ddof=1, axis=0)
    nonzero_var_idxs = np.where(vars > 0)[0]
    gene_expression = gene_expression.iloc[:, nonzero_var_idxs]

    gene_names = list(gene_expression.columns)
   
    gene_expression = gene_expression.astype(float).reset_index(drop=True)

    return gene_expression, gene_names

def get_gene_idx(gene_expression, gene_names):
    idxs = []
    genes = []
    for idx, gene in enumerate(list(gene_expression.columns)):
        if gene in gene_names:
            idxs.append(idx) 
            genes.append(gene)

    return idxs, genes

def calc_CSCORE(data, top_gene_indices=None):
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
    adata.var_names_make_unique()
    adata.raw = adata

    # Compute CSCORE
    res = CSCORE(adata, top_gene_indices)

    res0 = res[0]
    res0 = (res0 + res0.T) / 2

    return res[0]

def preprocess_and_save_all(kept_mgs_levels):
    repo_dir = os.path.dirname(os.getcwd())
    raw_dataset_path = os.path.join(repo_dir, "dataset", "raw_dataset", 'aak81Dataset.csv')
    genes_81 = list(pd.read_csv(raw_dataset_path).drop(columns=['Unnamed: 0', 'sample_id', 'mgs_level']).columns)


    preprocessed_dir = os.path.join(repo_dir, "dataset", "preprocessed_dataset")

    os.makedirs(preprocessed_dir, exist_ok=True)
    gene_expression, gene_names = preprocess_data(kept_mgs_levels=kept_mgs_levels)

    idxs_81, genes_81 = get_gene_idx(gene_expression, genes_81)

    cscore_cov_81 = calc_CSCORE(gene_expression, idxs_81)

    gene_dict_81 = {
        'gene_names': genes_81,
        'cscore_cov': cscore_cov_81}

    # Save the dictionary with pickle
    path_81 = os.path.join(preprocessed_dir, 'top_81_genes.pkl')
    with open(path_81, 'wb') as f:
        pickle.dump(gene_dict_81, f)


    warnings.filterwarnings("ignore")
    random.seed(42)
    random_gene_partitons = [random.sample(gene_names, 81) for _ in range(100)]

    for idx, genes in enumerate(random_gene_partitons):
        idxs, genes = get_gene_idx(gene_expression, genes)

        cscore_cov = calc_CSCORE(gene_expression, idxs)

        gene_dict = {
            'gene_names': genes,
            'cscore_cov': cscore_cov}

        # Save the dictionary with pickle
        path_random = os.path.join(preprocessed_dir, f'random_samples_{idx}.pkl')
        with open(path_random, 'wb') as f:
            pickle.dump(gene_dict, f)

def create_affinity_matrix(embeddings, P=3, S=7):
    n_samples = embeddings.shape[0]
    
    # Convert distances to similarities using a Gaussian kernel
    distances = euclidean_distances(embeddings)

    # For each point, find the distance to its k_scale-th neighbor
    local_sigmas = np.zeros(n_samples)
    for i in range(n_samples):
        # Sort distances and take the k_scale-th neighbor (k_scale=7 recommended by paper)
        sorted_dists = np.sort(distances[i])
        # k_scale+1 because the first distance is to itself (0)
        if P < len(sorted_dists):
            local_sigmas[i] = sorted_dists[P]
        else:
            # If not enough neighbors, use the furthest one
            local_sigmas[i] = sorted_dists[-1]

    #finding the S neights neighbors of each node 
    knn_graph = kneighbors_graph(embeddings, n_neighbors=S, mode='connectivity', include_self=False)
    knn_matrix = knn_graph.toarray().astype(bool)
    neighbor_sets = [set(np.where(knn_matrix[i])[0]) for i in range(n_samples)]

    # Use the local scaling approach: exp(-d(i,j)²/(sigma_i * sigma_j))
    affinity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            # Skip self-connections (diagonal)
            if i != j:
                #intersection points
                common_neighbors = neighbor_sets[i].intersection(neighbor_sets[j])
                #num of intersection points 
                cnn_count = len(common_neighbors)
                det = (local_sigmas[i] * local_sigmas[j]) * (cnn_count + 1)


                # Use the product of local sigmas as in the paper
                num = -(distances[i, j]**2)
                affinity_matrix[i, j] = np.exp(num / det)
        
    return affinity_matrix

def multimodality_method(A, max_k=20, c_max=7, f=2.0):
    """
    Implementation of the actual multimodality gap method from the paper.
    This analyzes eigenvector multimodality using dip test statistics.
    
    NOTE: This requires the diptest package: pip install diptest
    """
    
    # Build normalized Laplacian
    L = csgraph.laplacian(A, normed=True)
    
    # Get eigenvectors
    eigenvalues, eigenvectors = LA.eig(L)
    idx = np.argsort(np.real(eigenvalues))
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate dip test statistics Z = {z1, z2, ..., zN}
    Z = []
    for i in range(min(max_k, eigenvectors.shape[1])):
        eigvec = np.real(eigenvectors[:, i])
        dip_stat, _ = diptest(eigvec)
        Z.append(dip_stat)
    
    Z = np.array(Z)
    
    # Calculate differences D = {d1, d2, ..., dm}
    # d_i = z_i - z_{i-1}, starting from i=2 (index 1)
    D = np.diff(Z) 
    
    # Find last substantial multimodality gap
    if len(D) < 2:
        return 2
    
    # Skip d1 (non-informative), start with d2 as d_min
    d_min = D[1]
    d_min_idx = 2
    c = 1
    
    # Iterate through remaining differences
    for i in range(2, len(D)): 
        if (d_min / f) > D[i]:
            d_min = D[i]
            d_min_idx = i + 1
            c = 1 
        else:
            c += 1
            
        if c > c_max:
            break
    
    #k_star = i - 1 but we are starting at from range 2 to len(D) where paper starts at d_3 to d_max_k-1, therefore we do now need to subtract the -1 
    k_star = max(2, d_min_idx)
    
    return k_star


def covariance_matrix_to_graph(cov_matrix, threshold=0.7, random_partition=False):
    G = nx.Graph()
    n = cov_matrix.shape[0]

    # Add nodes
    for i in range(n):
        G.add_node(i)

    
    for i in range(n):
        for j in range(i + 1, n):
            weight = cov_matrix[i, j]
            if abs(weight) >= threshold: 
                G.add_edge(i, j, weight=weight)

    if not random_partition:
        isolated = list(nx.isolates(G))
        if isolated:
            raise ValueError(f"Graph has {len(isolated)} isolated nodes — cannot generate embeddings.")
    else: 
        pass

    return G

def make_walks(graph, p, q, workers, gamma, num_walks, walk_len):
    adj_mat = nx.to_numpy_array(graph)
    IDs = list(graph.nodes)

    #executing the walk Node2vecPlusPlus procedure 
    model = Node2vecPlusPlus.from_mat(adj_mat, IDs, p=p, q=q, workers=workers, gamma=gamma)
    walks = model.simulate_walks(num_walks=num_walks, walk_length=walk_len)

    return walks 

def generate_embeddings(model, nodes):
    embeddings = []
    for node in nodes:
        embeddings.append(model.wv[node])

    return np.vstack(embeddings) 

def plot_embeddings(embeddings, labels, gene_names):
    df = pd.DataFrame(embeddings, columns=['x', 'y'])
    df['label'] = labels
    df['gene_name'] = gene_names

    fig = px.scatter(df, x='x', y='y', color='label', hover_data=['gene_name'], 
                     title='Embedding Clustering Visualization', 
                     color_continuous_scale='Turbo')
    
    fig.update_layout(
        xaxis_title='Embedding Dimension 1',
        yaxis_title='Embedding Dimension 2',
        legend_title_text='Cluster Label',
        height=800,
        width=800
    )

    return fig

def plot_embeddings_3d(embeddings, labels, gene_names, symbol_map=None):
    df = pd.DataFrame(embeddings, columns=['x', 'y', 'z'])
    df['label'] = labels
    df['gene_name'] = gene_names

    # Convert label to string to ensure it's treated as categorical
    df['label'] = df['label'].astype(str)

    fig = px.scatter_3d(df, x='x', y='y', z='z', 
                     color='label', 
                     hover_data=['gene_name'], 
                     title='Embedding Clustering Visualization',
                     color_discrete_sequence=px.colors.qualitative.Set2,  
                     symbol=symbol_map if symbol_map is not None else None)
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Embedding Dimension 1'),
            yaxis=dict(title='Embedding Dimension 2'),
            zaxis=dict(title='Embedding Dimension 3'),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        legend_title_text='Cluster Label',
        height=800,
        width=800
    )

    fig.update_traces(
        marker=dict(
            size=6,
            opacity=0.7
        )
    )

    return fig


def run_job_full_dim(conv_matrix, edge_threshold, walk_hyperparams, kernel_hyperparams, model_hyperparams, gene_names, save_clusters=True, symbol_map=None, random_partition=False):
    try:
        graph = covariance_matrix_to_graph(conv_matrix, edge_threshold, random_partition=random_partition)
    except ValueError as e:
        return None, None, None

    #walks 
    walks = make_walks(graph, walk_hyperparams['p'], walk_hyperparams['q'], walk_hyperparams['workers'], walk_hyperparams['gamma'], walk_hyperparams['num_walks'], walk_hyperparams['walk_length'])
    
    #using the walks to train a vector for each 
    model = Word2Vec(walks, vector_size=model_hyperparams['embedding_dim'], window=model_hyperparams['window'], min_count=model_hyperparams['min_count'], sg=1, workers=walk_hyperparams['workers'], negative=model_hyperparams['negative'], seed=42, epochs=100)
    
    embeddings = generate_embeddings(model, range(graph.number_of_nodes()))


    affinity_matrix = create_affinity_matrix(embeddings, P=kernel_hyperparams["P"], S=kernel_hyperparams["S"])
    optimal_n_clusters = multimodality_method(affinity_matrix, max_k=6)
   
    spectral_clustering = SpectralClustering(n_clusters=optimal_n_clusters, eigen_solver='arpack', affinity='precomputed', assign_labels='cluster_qr',  random_state=42) 
    spectral_clustering.fit(affinity_matrix)
    spectral_labels = spectral_clustering.labels_

    tsne_embeddings = None
    if embeddings.shape[1] > 3:
        reducer = umap.UMAP(n_components=3, random_state=42, n_jobs=1)
        tsne_embeddings = reducer.fit_transform(embeddings)

    if tsne_embeddings is not None and tsne_embeddings.shape[1] == 2:
        fig2 = plot_embeddings(tsne_embeddings, spectral_labels, gene_names)
        fig2.update_layout(title=f"Cluster Type: Spectral, Embedding Dim {model_hyperparams['embedding_dim']}") 
    elif tsne_embeddings is not None and tsne_embeddings.shape[1] == 3:
        fig2 = plot_embeddings_3d(tsne_embeddings, spectral_labels, gene_names, symbol_map)
        fig2.update_layout(title=f"Cluster Type: Spectral, Embedding Dim {model_hyperparams['embedding_dim']}") 
    elif embeddings.shape[1] == 3:
        fig2 = plot_embeddings_3d(embeddings, spectral_labels, gene_names, symbol_map)
        fig2.update_layout(title=f"Cluster Type: Spectral, Embedding Dim {model_hyperparams['embedding_dim']}") 
    elif embeddings.shape[1] == 2:
        fig2 = plot_embeddings(embeddings, spectral_labels, gene_names)
        fig2.update_layout(title=f"Cluster Type: Spectral, Embedding Dim {model_hyperparams['embedding_dim']}")
    else:
        raise ValueError(f"Cannot plot embeddings with shape {embeddings.shape}")


    df = pd.DataFrame({
        'Gene Names': gene_names,
        'spectral_labels': spectral_labels,
        'embeddings': [embeddings[i] for i in range(len(embeddings))],
    })
    
    spectral_DBCVI = float(ClusteringMetric(X=embeddings, y_pred=spectral_labels).DBCVI())

    del model 

    return spectral_DBCVI, df, fig2

def objective(trial, data):
    edge_threshold = trial.suggest_float("edge_threshold", 0.3, 0.5, step=0.05)
    p = trial.suggest_float("p", 0.01, 100.0, log=True)
    q = trial.suggest_float("q", 0.01, 100.0, log=True)

    walk_length = trial.suggest_int("walk_length", 10, 30, step=5)
    P = trial.suggest_int("P", 3, 7, step=2)
    S = trial.suggest_int("S", P+2, P+4, step=2)

    #sampling form uniform 
    embedding_dim = trial.suggest_int("embedding_dim", 2, 16, step=2)
    window = trial.suggest_int("window", 4, 10, step=3)
    negative = trial.suggest_int("negative", 5, 15, step=2)

    walk_hyperparams = {
        "p": p,
        "q": q,
        "workers": 20,
        "gamma": 0,
        "num_walks": 256,
        "walk_length": walk_length
    }

    kernel_hyperparams = {
        "P": P,
        "S": S
    }

    model_hyperparams = {
        'embedding_dim': embedding_dim,
        'window': window,
        'min_count': 0,
        'negative': negative,
    }

    gene_names = data['gene_names']
    cscore_cov = data['cscore_cov']

    spectral_DBCVI_lst = []
    for i in range(8):
        spectral_DBCVI, _, _ = run_job_full_dim(cscore_cov, edge_threshold, walk_hyperparams, kernel_hyperparams, model_hyperparams, gene_names)
        if spectral_DBCVI is not None:
            spectral_DBCVI_lst.append(spectral_DBCVI)

        gc.collect()

    if spectral_DBCVI_lst:
        spectral_DBCVI_mean = s.mean(spectral_DBCVI_lst)
    else:
        spectral_DBCVI_mean = 0

    gc.collect()

    return spectral_DBCVI_mean

def optimize_hyperparameters(gene_dict, objective, n_trials):
    #HYPER OPTIMIZATION##
    # Creating the study
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    # Passing the function we are going to topimze 
    objective_with_path = partial(objective, data=gene_dict)

    study.optimize(objective_with_path, n_trials=n_trials)
    best_trial = study.best_trial
    best_params = best_trial.params
    print(best_trial)
    print(best_params) 

    return best_params
    #HYPER OPTIMIZATION##

def cluster_genes(best_params, gene_dict, results_path):
    os.makedirs(results_path, exist_ok=True)

    edge_threshold = best_params['edge_threshold']

    walk_hyperparams = {
        "p": best_params['p'],
        "q": best_params['q'],
        "workers": 24,
        "gamma": 0,
        "num_walks": 32768,
        "walk_length": best_params['walk_length']
    }
    kernel_hyperparams = {
        "P": best_params["P"],
        "S": best_params["S"],
    }
    model_hyperparams = {
        'embedding_dim': best_params['embedding_dim'],
        'window': best_params['window'],
        'min_count': 0,
        'negative': best_params['negative'],
    }

    spectral_DBCVI_lst = []
    for i in tqdm(range(0, 100)):
        spectral_DBCVI, df, fig =  run_job_full_dim(gene_dict['cscore_cov'], edge_threshold, walk_hyperparams, kernel_hyperparams, model_hyperparams, gene_dict['gene_names'])
        spectral_DBCVI_lst.append(spectral_DBCVI)
        print(spectral_DBCVI)
        # df.to_excel(os.path.join(results_path, f'clustering_trial_{i}.xlsx'), index=False)
        # fig.write_html(os.path.join(results_path, f'clustering_trial_{i}.html'))
        # gc.collect()

    with open(os.path.join(results_path, 'spec_dbcvi.json'), 'w+') as file:
        json.dump(spectral_DBCVI_lst, file)

def cluster_random_genes(best_params, results_path):
    warnings.filterwarnings("ignore")
    os.makedirs(results_path, exist_ok=True)

    edge_threshold = best_params['edge_threshold']

    walk_hyperparams = {
        "p": best_params['p'],
        "q": best_params['q'],
        "workers": 24,
        "gamma": 0,
        "num_walks": 32768,
        "walk_length": best_params['walk_length']
    }
    kernel_hyperparams = {
        "P": best_params["P"],
        "S": best_params["S"],
    }
    model_hyperparams = {
        'embedding_dim': best_params['embedding_dim'],
        'window': best_params['window'],
        'min_count': 0,
        'negative': best_params['negative'],
    }

    repo_dir = Path.cwd().parent
    preprocess_data_dir = os.path.join(repo_dir, 'dataset', 'preprocessed_dataset')

    spectral_DBCVI_lst = []
    for idx in tqdm(range(0, 100)):

        random_path = os.path.join(preprocess_data_dir, f'random_samples_{idx}.pkl')

        with open(random_path, 'rb') as f:
            genes_dict = pickle.load(f)

        spectral_DBCVI, df, fig =  run_job_full_dim(genes_dict['cscore_cov'], edge_threshold, walk_hyperparams, kernel_hyperparams, model_hyperparams, genes_dict['gene_names'], random_partition=True)
        spectral_DBCVI_lst.append(spectral_DBCVI)
        df.to_excel(os.path.join(results_path, f'clustering_trial_{idx}.xlsx'), index=False)
        fig.write_html(os.path.join(results_path, f'clustering_trial_{idx}.html'))
        gc.collect()

    with open(os.path.join(results_path, 'spec_dbcvi.json'), 'w+') as file:
        json.dump(spectral_DBCVI_lst, file)

def stats_pvalue(gene_spectral_lst, random_gene_spectral_lst):

    ks_res = stats.ks_2samp(gene_spectral_lst, random_gene_spectral_lst)
    ad_res = stats.anderson_ksamp([gene_spectral_lst, random_gene_spectral_lst])

    print(f"KS P Value: {ks_res.pvalue}")
    print(f"AD P Value: {ad_res.pvalue}")

def compute_AMI(results_path):
    # Load all clustering result DataFrames
    dfs = [pd.read_excel(os.path.join(results_path, f'clustering_trial_{i}.xlsx'))
        for i in tqdm(range(100), desc='Loading data')]
    # Compute Adjusted Mutual Information (AMI)
    ami_lst = []

    for i in tqdm(range(100), desc='Computing AMI'):
        labels_i = dfs[i]['spectral_labels']
        for j in range(i + 1, 100):
            labels_j = dfs[j]['spectral_labels']
            ami = adjusted_mutual_info_score(labels_i, labels_j)
            ami_lst.append(ami)
    print(f'AMI: {s.mean(ami_lst)}')

# Function to plot KDE and include vertical lines
def plot_kde(spectral_81_lst, random_lst):
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(spectral_81_lst, shade=True, label=f'AMD Genes DBCVI Distribution', color='red', bw_adjust=1.5)
    sns.kdeplot(random_lst, shade=True, label=f'Random Genes DBCVI Distribution', color='blue', bw_adjust=1.5)
    
    plt.xlabel('DBCVI Value', fontsize=20)
    plt.ylabel('Probability Density', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis='y', alpha=0.75)
    plt.legend(fontsize=20)
    
    plt.tight_layout()
    plt.savefig("kde_plot.pdf")
    plt.show()
