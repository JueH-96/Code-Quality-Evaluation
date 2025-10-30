def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M, K = map(int, input_data[:3])
    edges_raw = input_data[3:]
    edges = []
    idx = 0
    for _ in range(M):
        u = int(edges_raw[idx]); v = int(edges_raw[idx+1]); w = int(edges_raw[idx+2])
        idx += 3
        # Store edges as (u-1, v-1, weight) for 0-based union-find
        edges.append((u-1, v-1, w))
    
    # A standard union-find (disjoint set) for checking connectivity
    def find(parent, x):
        if parent[x] != x:
            parent[x] = find(parent, parent[x])
        return parent[x]
    
    def union(parent, rank, x, y):
        rx = find(parent, x)
        ry = find(parent, y)
        if rx != ry:
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
    
    from itertools import combinations
    
    min_cost = K  # Since weights < K, the cost mod K ranges up to K-1
    indices = range(M)
    
    # Check all combinations of edges of size N-1
    for combo in combinations(indices, N-1):
        parent = list(range(N))
        rank = [0]*N
        s = 0
        # Union edges in this combination, compute sum mod K
        for c in combo:
            u, v, w = edges[c]
            s = (s + w) % K
            union(parent, rank, u, v)
        
        # Check if exactly one connected component
        root_set = set(find(parent, i) for i in range(N))
        if len(root_set) == 1:
            if s < min_cost:
                min_cost = s
    
    print(min_cost)

if __name__ == "__main__":
    main()