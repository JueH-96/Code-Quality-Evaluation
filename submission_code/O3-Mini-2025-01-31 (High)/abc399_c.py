def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    
    # Initialize Union-Find (Disjoint Set Union) structures.
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    
    # Find with path compression.
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    # Union by rank.
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return False  # Already in the same component.
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        else:
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
        return True
    
    # Process each edge.
    index = 2
    for _ in range(M):
        u = int(data[index])
        v = int(data[index + 1])
        index += 2
        union(u, v)
        
    # Count the number of connected components.
    components = 0
    for i in range(1, N + 1):
        if find(i) == i:
            components += 1
            
    # In a forest, each tree (connected component) with k nodes has exactly k-1 edges.
    # Hence, a spanning forest on an N-vertex graph with `components` connected components
    # must have exactly (N - components) edges. The extra (M - (N - components)) edges
    # cause cycles. Thus they need to be removed.
    answer = M - (N - components)
    sys.stdout.write(str(answer))
    
if __name__ == "__main__":
    main()