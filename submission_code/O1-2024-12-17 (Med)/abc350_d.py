def main():
    import sys
    sys.setrecursionlimit(10**7)

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    edges = data[2:]
    
    # Union-Find (Disjoint Set Union) implementation
    parent = list(range(N+1))
    size = [1] * (N+1)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

    idx = 0
    for _ in range(M):
        a = int(edges[idx]); b = int(edges[idx+1])
        idx += 2
        union(a, b)

    # Count sizes of connected components
    component_sizes = {}
    for i in range(1, N+1):
        root_i = find(i)
        component_sizes[root_i] = component_sizes.get(root_i, 0) + 1
    
    # After repeated operations, each connected component becomes a complete subgraph.
    # The number of edges in a complete subgraph of size k is k*(k-1)//2.
    # We sum this up for all components and subtract the current number of edges M.
    total_complete_edges = 0
    for size_c in component_sizes.values():
        total_complete_edges += size_c * (size_c - 1) // 2
    
    result = total_complete_edges - M
    print(result)

if __name__ == "__main__":
    main()