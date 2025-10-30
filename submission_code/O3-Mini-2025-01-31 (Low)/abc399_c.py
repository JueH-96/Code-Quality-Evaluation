def main():
    import sys
    sys.setrecursionlimit(10**6)
    input = sys.stdin.readline
    
    # Read the number of vertices and edges
    N, M = map(int, input().split())
    
    # Build graph adjacency list (optional, we can also use union-find)
    # We'll use Union-Find (Disjoint Set Union) to determine number of connected components.
    
    parent = list(range(N + 1))
    rank = [0] * (N + 1)
    
    def find(x):
        # path compression
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False  # They are already in the same set
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        else:
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
        return True
    
    # Count connected components using union-find
    components = N
    for _ in range(M):
        u, v = map(int, input().split())
        if union(u, v):
            components -= 1
    
    # For a forest, we need exactly (N - components) edges.
    # Thus, the extra edges that have to be removed are:
    excess_edges = M - (N - components)
    # Output the result
    sys.stdout.write(str(excess_edges))

if __name__ == '__main__':
    main()