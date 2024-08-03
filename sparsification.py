import sympy as sp
from sympy import Matrix, latex
import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import matplotlib.pyplot as plt


# function to reload module in iPython code
def check():
    return


def norm(vector: Matrix) -> Matrix:
    return sp.sqrt(vector.dot(vector))


def normalized(vector: Matrix) -> Matrix:
    return vector / norm(vector)


def adjacency_to_laplacian(matrix: np.ndarray) -> Matrix:
    A = -Matrix(matrix)
    for i in range(A.shape[0]):
        A[i, i] = -sum(A.row(i))
    return A


def laplacian_to_adjacency(matrix: Matrix) -> Matrix:
    return -matrix + sp.diag(*matrix.diagonal())


def sympy_to_numpy(matrix: Matrix) -> np.ndarray:
    return np.array(matrix).astype(np.float64)


def build_tex(L, Phi_k, Phi_gr_k, Lambda_k, F, Sp):
    result = f"""
    A = {latex(laplacian_to_adjacency(L))} \\\\
    L = {latex(L)} \\\\
    \\Phi_k = {latex(Phi_k)} \\\\
    \\Phi_{{>k}} = {latex(Phi_gr_k)} \\\\
    \\Lambda_k = {latex(Lambda_k)} \\\\
    F = {latex(F)} \\\\
    Sp_G(k): \\\\
    """

    for i, matrix in enumerate(Sp):
        result += f"L_{{{i + 1}}} = {latex(Matrix(matrix))} \\\\"

    with open("res.tex", 'w', encoding='utf8') as f:
        f.write(result)


def draw_result(L, Sp):
    for i, matrix in enumerate([laplacian_to_adjacency(L)] + Sp):
        G = nx.from_numpy_array(sympy_to_numpy(matrix))

        plt.figure(i)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)

        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def sparsify(
        graph: nx.Graph | nx.DiGraph | np.ndarray | Matrix | str,
        k: int = 2,
        n: int = 10,
        trivial: bool = False,
        return_sympy: bool = False,
        log: bool = False,
        show: bool = False
) -> list[np.ndarray] | list[Matrix]:
    """
    Performs an isospectral sparsification of a graph.

    Parameters
    ----------
    graph : networkx.Graph|networkx.DiGraph|numpy.ndarray|sympy.Matrix|str
        Graph that needs to be sparsified.
        networkx.Graph|networkx.DiGraph -- graph, that needs to be sparsified;
        numpy.ndarray|sympy.Matrix -- adjacency matrix of a graph;
        str -- path to csv-file, where adjacency matrix of a graph is stored
    k : int, optional (default=2)
        Number of eigenvalues to be preserved from initial graph
    n : int, optional (default=10)
        Number of generated sparsified graphs
    trivial : bool, optional (default=False)
        If set to True, generate trivial results with zero and identity matrices
    return_sympy : bool, optional (default=False)
        If set to True, return the sparsified graph as a sympy matrix
    log : bool, optional (default=False)
        If set to True, log the result to console
    show : bool, optional (default=False)
        If set to True, show the plot of initial graph, then 'n' sparsified graphs

    Returns
    -------
    list[numpy.ndarray]
        A list of adjacency matrices of 'n' sparsified graphs

    Raises
    ------
    TypeError
        when graph type is not supported
    ValueError
        when 'k' or 'n' are out of bounds
    """

    L = Matrix()
    if isinstance(graph, str):
        data_frame = pd.read_csv(graph, sep=',', header=None)
        L = adjacency_to_laplacian(np.array(data_frame.values))
    elif isinstance(graph, np.ndarray) or isinstance(graph, Matrix):
        L = adjacency_to_laplacian(np.array(graph))
    elif isinstance(graph, nx.Graph) or isinstance(graph, nx.DiGraph):
        L = Matrix(nx.laplacian_matrix(graph).toarray())
    else:
        raise TypeError(f"Graph type {type(graph)} is not supported.")

    if not (1 < k < L.shape[0]):
        raise ValueError(f"k should be in interval [2; n - 1]. k = {k} is out of [2; {L.shape[0] - 1}]")
    if n <= 0:
        raise ValueError(f"n = {n} should be positive.")

    eigenvals = []
    eigenvects = []

    for triple in L.eigenvects():
        for vector in triple[2]:
            eigenvals.append(triple[0])
            eigenvects.append(vector)

    Phi_k = Matrix.hstack(*[normalized(v) for v in eigenvects[:k]])
    Phi_gr_k = Matrix.hstack(*[normalized(v) for v in eigenvects[k:]])

    Lambda_k = sp.diag(*eigenvals[:k])

    F = Phi_k * Lambda_k * Phi_k.T + eigenvals[k - 1] * Phi_gr_k * Phi_gr_k.T

    df = len(eigenvals) - k
    Ys = [sp.diag(*[rd.randint(0, 10) for _ in range(df)]) for _ in range(n)] \
        if not trivial \
        else [sp.zeros(df), sp.eye(df)]

    Sp = []
    for Y in Ys:
        result = F + Phi_gr_k * Y * Phi_gr_k.T
        Sp.append(laplacian_to_adjacency(result))

    if log:
        build_tex(L, Phi_k, Phi_gr_k, Lambda_k, F, Sp)

    if show:
        draw_result(L, Sp)

    if return_sympy:
        return Sp

    return [sympy_to_numpy(x) for x in Sp]
