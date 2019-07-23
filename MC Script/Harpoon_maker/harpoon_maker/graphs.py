import numpy as np
import networkx as nx
from networkx.generators.classic import empty_graph, path_graph, complete_graph


def directed_gnm_random_graph(n, m, seed=None):
    """Returns a $G_{n,m}$ random graph.

    In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
    of all graphs with $n$ nodes and $m$ edges.

    This algorithm should be faster than :func:`dense_gnm_random_graph` for
    sparse graphs.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.
    seed : int, optional
        Seed for random number generator (default=None).

    See also
    --------
    dense_gnm_random_graph

    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.name = "gnm_random_graph(%s,%s)" % (n, m)

    if seed is not None:
        np.random.seed(seed)

    if n == 1:
        return G
    
    max_edges=n*(n-1)
    if m>=max_edges:
        return complete_graph(n,create_using=G)

    nlist = G.nodes()
    edge_count = 0
    while edge_count < m:
        # generate random edge,u,v
        u = np.random.choice(nlist)
        v = np.random.choice(nlist)
        if u==v or G.has_edge(u, v):
            continue
        else:
            G.add_edge(u, v)
            edge_count = edge_count+1
    return G


def _random_subset(seq,m):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets = set()
    while len(targets)<m:
        x = np.random.choice(seq)
        targets.add(x)
    return targets


def powerlaw_graph(n, m, seed=None):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1 and m < n, m = %d, n = %d" % (m, n))
    if seed is not None:
        np.random.seed(seed)

    # Add m initial nodes (m0 in barabasi-speak)
    G = empty_graph(m, create_using=nx.DiGraph())
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        targets = _random_subset(repeated_nodes, m)
        source += 1
    return G


def bi_powerlaw_graph(n, m, seed=None):
    """Returns a random graph according to the Barabási–Albert preferential
    attachment model.

    A graph of $n$ nodes is grown by attaching new nodes each with $m$
    edges that are preferentially attached to existing nodes with high degree.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Seed for random number generator (default=None).

    Returns
    -------
    G : Graph

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m < n``.

    References
    ----------
    .. [1] A. L. Barabási and R. Albert "Emergence of scaling in
       random networks", Science 286, pp 509-512, 1999.
    """

    # if m < 1 or m >= n:
    #     raise nx.NetworkXError("Barabási–Albert network must have m >= 1 and m < n, m = %d, n = %d" % (m, n))
    if seed is not None:
        np.random.seed(seed)

    n = 10
    div = np.random.randint(0, n)
    g1_dict = nx.to_dict_of_lists(powerlaw_graph(n, 1, seed=None))
    g1r_dict = nx.to_dict_of_lists(powerlaw_graph(n - 1, 1, seed=None).reverse())

    for i in range(len(g1r_dict)):
        g1_dict[i].extend(g1r_dict[i])

    G = nx.to_networkx_graph(g1_dict, create_using=nx.DiGraph())

    return G
