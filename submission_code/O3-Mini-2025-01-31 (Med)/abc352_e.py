def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # We will collect candidate edges in a dictionary.
    # In an operation, we are given a sorted list S = [v1, v2, ..., vk] and a cost C.
    # The complete graph on these vertices has many edges but its spanning tree can be
    # given by the "chain" [ (v1, v2), (v2, v3), ..., (v(k-1),vk) ].
    # Consequently, for each operation we add these chain edges (and if an edge already appears,
    # we keep the minimum weight).
    edges = {}
    for _ in range(m):
        k = int(next(it))
        c = int(next(it))
        vertices = [int(next(it)) for _ in range(k)]
        for i in range(k - 1):
            u = vertices[i]
            v = vertices[i + 1]  # by input guarantee, u < v.
            key = (u, v)
            if key in edges:
                if c < edges[key]:
                    edges[key] = c
            else:
                edges[key] = c
    
    # Prepare the list of edges as tuples (weight, u, v).
    edge_list = [(w, u, v) for (u, v), w in edges.items()]
    edge_list.sort(key=lambda x: x[0])
    
    # Implement Disjoint Set Union structure.
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    def find(x):
        # Path compression.
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False
        # Union by rank.
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
        return True
    
    total_weight = 0
    used_edges = 0
    # Standard Kruskal's algorithm.
    for w, u, v in edge_list:
        if union(u, v):
            total_weight += w
            used_edges += 1
            if used_edges == n - 1:
                break
    
    # If we have exactly n-1 edges in the MST, the graph is connected
    if used_edges == n - 1:
        sys.stdout.write(str(total_weight))
    else:
        sys.stdout.write(str(-1))
    
if __name__ == '__main__':
    main()