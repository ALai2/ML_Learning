import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from itertools import chain
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy import linalg

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import rbf_kernel

import datasets as dt

# generate graph laplacian with k nearest neighbors metric
def generate_graph_laplacian(df, nn):
    # Adjacency and Diagonal Matrices.
    adjacency_matrix = kneighbors_graph(X=df, n_neighbors=nn).toarray()
    diagonal_matrix = np.diag(adjacency_matrix.sum(axis=1))
    
    # Graph Laplacian.
    graph_laplacian = diagonal_matrix - adjacency_matrix
    return graph_laplacian 

# generate graph laplacian with rbf kernel function
def generate_graph_laplacian_with_rbf(df, gamma):
    # Adjacency Matrix.
    #connectivity = pairwise_kernels(df, metric='rbf')
    adjacency_matrix = rbf_kernel(df, gamma=gamma)
    diagonal_matrix = np.diag(adjacency_matrix.sum(axis=1))

    # Graph Laplacian.
    graph_laplacian = diagonal_matrix - adjacency_matrix
    return graph_laplacian 

# compute eigenvalues and eigenvectors from graph laplacian
def compute_eigen_pairs(graph_laplacian):
    eigenvals, eigenvcts = linalg.eig(graph_laplacian)

    # linalg gives back complex values, need real values
    eigenvals = np.real(eigenvals)
    eigenvcts = np.real(eigenvcts)
    return eigenvals, eigenvcts

# select eigenvectors by sorting eigenvalues
def select_eigenvectors(eigenvals, eigenvcts, num_ev):
    eigenvals_sorted_indices = np.argsort(eigenvals)
    indices = eigenvals_sorted_indices[: num_ev]

    proj_df = pd.DataFrame(eigenvcts[:, indices.squeeze()])
    proj_df.columns = ['v_' + str(c) for c in proj_df.columns]
    return proj_df

# use k-means clustering to cluster using eigenvectors
def run_k_means(df, n_clusters):
    k_means = KMeans(random_state=25, n_clusters=n_clusters)
    k_means.fit(df)
    cluster = k_means.predict(df)
    return cluster

# run the spectral clustering algorithm with selected parameters
def spectral_clustering(df, n_neighbors, n_clusters, gamma, use_neighbors=True):
    if use_neighbors:
        graph_laplacian = generate_graph_laplacian(df, n_neighbors)
    else:
        graph_laplacian = generate_graph_laplacian_with_rbf(df, gamma)
    eigenvals, eigenvcts = compute_eigen_pairs(graph_laplacian)
    proj_df = select_eigenvectors(eigenvals, eigenvcts, n_clusters)
    cluster = run_k_means(proj_df, proj_df.columns.size)
    return ['c_' + str(c) for c in cluster]

# plot the clusters selected by the spectral clustering algorithm
def plot_spectral_clusters(data_df):
    data_df['cluster'] = spectral_clustering(df=data_df, n_neighbors=8, n_clusters=2, gamma=450, use_neighbors=False)
    # noise=.05 for n_neighbors
    # gamma=175 for make_circles noise=.08
    # noise in dataset makes bad clusters, best clustering with noise=1 is with gamma=450
    # gamma=25 for moons

    fig, ax = plt.subplots()
    sns.scatterplot(x='x', y='y', data=data_df, hue='cluster', ax=ax)
    ax.set(title='Spectral Clustering')
    plt.show()

# plot the first ten eigenvalues of dataset's graph laplacian
def show_eigenvalues_plot(data_df, clusterNum):
    graph_laplacian = generate_graph_laplacian(df=data_df, nn=8)
    eigenvals, eigenvcts = compute_eigen_pairs(graph_laplacian)
    eigenvals_sorted_indices = np.argsort(eigenvals)
    eigenvals_sorted = eigenvals[eigenvals_sorted_indices]

    index_lim = 10
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=range(1, eigenvals_sorted_indices[: index_lim].size + 1), y=eigenvals_sorted[: index_lim], s=80, ax=ax)
    sns.lineplot(x=range(1, eigenvals_sorted_indices[: index_lim].size + 1), y=eigenvals_sorted[: index_lim], alpha=0.5, ax=ax)
    ax.axvline(x=clusterNum, color=sns_c[3], label='zero eigenvalues', linestyle='--')
    ax.legend()
    ax.set(title=f'Sorted Eigenvalues Graph Laplacian (First {index_lim})', xlabel='index', ylabel=r'$\lambda$')
    plt.show()

# generate data
#data_df = dt.data_frame_concentric_circles()
#data_df = dt.data_frame_from_moons()
data_df = dt.data_frame_make_circles()

# use data
plot_spectral_clusters(data_df)
#show_eigenvalues_plot(data_df, 2)