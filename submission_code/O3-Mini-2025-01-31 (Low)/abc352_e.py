def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # When adding a complete subgraph among a set S of vertices, note that a chain (consecutive pairs)
    # is enough for constructing the MST.  That is, for S sorted, if we add edges (A1, A2), (A2, A3), …,
    # (A_{k-1}, A_k) with the given weight W, then the clique is "represented" for MST purposes.
    # So we only need to consider these consecutive pairs.
    # However, if the same edge appears more than once (over different operations or same operation in different pairings)
    # we only want to record the smallest weight.
    edges = {}
    # Process each operation
    for _ in range(m):
        k = int(next(it))
        w = int(next(it))
        vertices = [int(next(it)) for _ in range(k)]
        for i in range(k - 1):
            u = vertices[i]
            v = vertices[i+1]
            # (u,v) are already in sorted order because A_i's are in increasing order.
            key = (u, v)
            if key in edges:
                if w < edges[key]:
                    edges[key] = w
            else:
                edges[key] = w

    # Prepare all candidate edges for MST from the operations.
    # Each edge is a tuple (u,v,w)
    candidate_edges = [(u, v, weight) for (u, v), weight in edges.items()]
    
    # Sort edges by weight. (Kruskal's algorithm)
    candidate_edges.sort(key=lambda x: x[2])
    
    # Disjoint set data structure for Kruskal's
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True
    
    total_weight = 0
    edge_count = 0
    for u, v, w in candidate_edges:
        if union(u, v):
            total_weight += w
            edge_count += 1
            if edge_count == n - 1:
                break
    
    # Check if the graph becomes connected.
    # We can check if all nodes share the same root.
    root = find(1)
    for i in range(2, n + 1):
        if find(i) != root:
            sys.stdout.write("-1")
            return
    sys.stdout.write(str(total_weight))
    
if __name__ == '__main__':
    main()