def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # We use the observation that each operation adds a complete graph (a clique)
    # on the vertices in S with all edges having the same weight C.
    # For computing the MST, it is sufficient to include a spanning tree for each clique.
    # A simple spanning tree of S is obtained by connecting consecutive vertices in S,
    # because the vertices in S are given sorted (A_{i,1} < A_{i,2} < ...).
    # So for each operation, we add the edge (S[j], S[j+1]) with cost C.
    
    edges = []
    for _ in range(m):
        k = int(next(it))
        cost = int(next(it))
        S = [int(next(it)) for _ in range(k)]
        for i in range(k - 1):
            edges.append((cost, S[i], S[i + 1]))
    
    edges.sort(key=lambda x: x[0])
    
    # Implementation of union-find (Disjoint Set Union, DSU)
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    def find(x):
        # Iterative path compression
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
    added_edges = 0
    for w, u, v in edges:
        if union(u, v):
            total_weight += w
            added_edges += 1
            if added_edges == n - 1:  # Early stopping if MST is complete
                break

    # If we haven't added exactly n-1 edges, the graph is not connected.
    if added_edges < n - 1:
        sys.stdout.write("-1")
    else:
        sys.stdout.write(str(total_weight))

if __name__ == '__main__':
    main()