import networkx as nx
import random
import numpy as np
import scipy as sp
import copy
from scipy.sparse import csr_matrix
from typing import Dict, Tuple
from sklearn import datasets
from ot.datasets import make_1D_gauss as gauss


#%% Functions for generating graphs
def undirected_normalized_heat_kernel(G, t):
    # Input: Graph G and time parameter t
    # Output: heat kernel matrix
    # Automatically computes directed laplacian matrix and then exponentiates
    # Copied from github: trneedham/Spectral-Gromov-Wasserstein/blob/master/spectralGW.py

    L = nx.normalized_laplacian_matrix(G).toarray()
    lam, phi = np.linalg.eigh(L)
    heat_kernel = np.matmul(phi, np.matmul(np.diag(np.exp(-t*lam)), phi.T))
    return heat_kernel


def extract_graph_info(graph: nx.Graph, method: str = 'adjacency') -> Tuple[np.ndarray, csr_matrix, Dict]:
    """
    Extract node distribution, adjacency matrix, and node index dictionary from networkx graph structure
    Args:
        graph: the graph instance generated via networkx
        method: 'adjacency' or 'shortest_path' or 'heat_kernel'
        
    Returns:
        probs: the distribution of nodes
        adj: adjacency matrix
        idx2node: a dictionary {key: idx, value: node name}
    """
    idx2node = {}
    for i in range(len(graph.nodes)):
        idx2node[i] = i
    
    if method == 'adjacency':
        adj = nx.adjacency_matrix(graph)
        
    elif method == 'shortest_path':
        length_pair = dict(nx.all_pairs_shortest_path_length(graph))
        n_node = len(length_pair)
        adj = np.zeros((n_node, n_node))
        for ii in range(n_node):
            for jj in range(n_node):
                adj[ii,jj] = length_pair[ii][jj]
    
    else:
        adj = undirected_normalized_heat_kernel(graph, 1)
    
    degrees = np.array(list(graph.degree))[:, 1]
    probs = degrees / np.sum(degrees)
    return probs, csr_matrix(adj), idx2node


def add_noisy_nodes(graph: nx.graph, noisy_level: float) -> nx.graph:
    """
    Add noisy (random) nodes in a graph
    Args:
        graph: the graph instance generated via networkx
        noisy_level: the percentage of noisy nodes compared with original edges
    Returns:
        graph_noisy: the noisy graph
    """
    num_nodes = len(graph.nodes)
    num_noisy_nodes = int(noisy_level * num_nodes)

    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges / num_nodes + 1)

    graph_noisy = copy.deepcopy(graph)
    if num_noisy_nodes > 0:
        for i in range(num_noisy_nodes):
            graph_noisy.add_node(int(i + num_nodes))
            j = 0
            while j < num_noisy_edges:
                src = random.choice(list(range(i + num_nodes)))
                if (src, int(i + num_nodes)) not in graph_noisy.edges:
                    graph_noisy.add_edge(src, int(i + num_nodes))
                    j += 1
    return graph_noisy


def add_noisy_edges(graph: nx.graph, noisy_level: float) -> nx.graph:
    """
    Add noisy (random) edges in a graph
    Args:
        graph: the graph instance generated via networkx
        noisy_level: the percentage of noisy edges compared with original edges
    Returns:
        graph_noisy: the noisy graph
    """
    nodes = list(graph.nodes)
    num_edges = len(graph.edges)
    num_noisy_edges = int(noisy_level * num_edges)
    graph_noisy = copy.deepcopy(graph)
    if num_noisy_edges > 0:
        i = 0
        while i < num_noisy_edges:
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if (src, dst) not in graph_noisy.edges:
                graph_noisy.add_edge(src, dst)
                i += 1
    return graph_noisy


def graph_pair_simulator(n_nodes: int, clique_size: int, p_in: float = 0.4, p_out: float = 0.1,
                         g_type: str = 'brp', noise_level: float = 0.05, noise_type: str = 'edge',
                         method: str = 'adjacency'):
    """
    :param n_nodes: the number of nodes in the source graph
    :param clique_size: the number of nodes in a subgraph
    :param p_in: the probablity of edges within a subgrpah
    :param p_out: the probability of edges across different subgraphs
    :param g_type: the type of graph "powerlaw" or "grp"
    :param noise_level: the percentage of noisy edges in a graph.
    :param noise_type: adding noisy edges or adding noisy nodes and edges
    :param method: 'adjacency' or 'shortest_path' or 'heat_kernel'
    
    :return:
    """
    if g_type == 'powerlaw':
        graph_src = nx.powerlaw_cluster_graph(n=n_nodes, m=int(clique_size * p_in), p=p_out * clique_size / n_nodes)
    else:
        graph_src = nx.gaussian_random_partition_graph(n=n_nodes, s=clique_size, v=5,
                                                       p_in=p_in, p_out=p_out, directed=False)

    if noise_type == 'edge':
        graph_dst = add_noisy_edges(graph_src, noise_level)
    else:
        graph_dst = add_noisy_edges(graph_src, noise_level)
        graph_dst = add_noisy_nodes(graph_dst, noise_level)

    # weights = np.random.rand(num_nodes[nn], num_nodes[nn]) + 1
    p_s, cost_s, idx2node_s = extract_graph_info(graph_src, method)
    p_t, cost_t, idx2node_t = extract_graph_info(graph_dst, method)
    return p_s, p_t, cost_s, cost_t, idx2node_s, idx2node_t



#%% Functions for generating spriral-type data points
# Code of generating the SPIRAL dataset is copied from github: tvayer/SGW/blob/master/risgw_example.ipynb
def make_spiral(n_samples, noise):
    
    n = np.sqrt(np.random.rand(n_samples,1)) * (3*np.pi)
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    
    return np.array(np.hstack((d1x,d1y)))


get_rot = lambda theta : np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),np.cos(theta)]])


def get_spiral_data(n_samples, theta, scale, transla, noise):
    
    Xs = make_spiral(n_samples, noise) - transla
    Xt = make_spiral(n_samples, noise)
    
    A = get_rot(theta)
    Xt = (np.dot(Xt,A)) * scale + transla
    
    return Xs, Xt



#%% Summarized function for generating sythetic datasets
def data_generator(name: str, n_sample: int, method: str = 'adjacency'):
    """
    :param name: the name of synthetic dataset "moon", "graph", "gaussian", or "spiral"
    :param n_sample: the number of points/nodes in source or target samples
    :param method: 'adjacency' or 'shortest_path' or 'heat_kernel' (only work if name=='graph')
    
    :return: 
        source and target distributions (a, b)
        source and target relation matrices (C1, C2: np.array)
    """
    if name == 'moon':
        noisy_moons = datasets.make_moons(n_samples=2*n_sample, noise=0.1)
        xs = noisy_moons[0][noisy_moons[1]==0]
        xt = noisy_moons[0][noisy_moons[1]==1]
    
    elif name == 'graph':
        a, b, cost_s, cost_t, idx2node_s, idx2node_t = graph_pair_simulator(
            n_nodes=n_sample,
            clique_size=int(n_sample/5),
            p_in=0.2, 
            p_out=0.02,
            g_type='powerlaw',  # 'powerlaw' or 'grp'
            noise_level = 0.2,
            noise_type='edge',  # 'edge' or 'edge+node'
            method=method)
        
        a = a**3
        b = b**3
        a /= np.sum(a)
        b /= np.sum(b)
        
        C1 = cost_s.toarray().astype(float)
        C2 = cost_t.toarray().astype(float)
    
    elif name == 'gaussian':
        ds = 5
        dt = 10
        mu1 = np.tile(0, ds)
        mu2 = np.tile(1, ds)
        mu3 = np.array([0,2,2,0,0])
        cov = np.eye(ds)
        for ii in range(ds):
            for jj in range(ds):
                cov[ii,jj] = 0.6**(abs(ii-jj))
        xs = np.random.multivariate_normal(mu1, cov, int(n_sample/3))
        xs = np.vstack([xs, np.random.multivariate_normal(mu2, cov, int(n_sample/3))]) 
        xs = np.vstack([xs, np.random.multivariate_normal(mu3, cov, n_sample-2*int(n_sample/3))])
        
        mu1 = np.tile(0.5, dt)
        mu2 = np.tile(2, dt)
        xt = np.random.multivariate_normal(mu1, np.eye(dt), int(n_sample/2))
        xt = np.vstack([xt, np.random.multivariate_normal(mu2, np.eye(dt), n_sample-int(n_sample/2))])

    elif name == 'spiral':
        xs, xt = get_spiral_data(n_sample, theta=np.pi/4, scale=1, transla=10, noise=1)

    
    if name != 'graph':
        a = gauss(n_sample, m=n_sample/3, s=n_sample/20)  # m=mean, s=std
        b = gauss(n_sample, m=n_sample/2, s=n_sample/20)  # m=mean, s=std
        
        C1 = sp.spatial.distance.cdist(xs, xs)
        C2 = sp.spatial.distance.cdist(xt, xt)
        C1 /= np.max(C1)
        C2 /= np.max(C2)
        
    return a, b, C1, C2
