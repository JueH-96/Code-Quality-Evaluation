def solve():
    n, k = map(int, input().split())
    edges = []
    for _ in range(n - 1):
        edges.append(tuple(map(int, input().split())))
    
    required_vertices = list(map(int, input().split()))

    if k == 1:
        print(1)
        return

    import networkx as nx
    
    graph = nx.Graph()
    graph.add_nodes_from(range(1, n + 1))
    graph.add_edges_from(edges)

    min_vertices = float('inf')

    for i in range(1 << n):
        subset_vertices = []
        for j in range(n):
            if (i >> j) & 1:
                subset_vertices.append(j + 1)
        
        valid_subset = True
        for v in required_vertices:
            if v not in subset_vertices:
                valid_subset = False
                break
        
        if not valid_subset:
            continue
        
        subgraph = graph.subgraph(subset_vertices)
        
        if nx.is_forest(subgraph):
            num_components = nx.number_connected_components(subgraph)
            if num_components == 1 or len(subset_vertices) == 1:
                min_vertices = min(min_vertices, len(subset_vertices))
        
    print(min_vertices)

solve()