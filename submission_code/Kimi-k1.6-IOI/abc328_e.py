import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    K = int(input[idx])
    idx += 1
    edges = []
    for _ in range(M):
        u = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        w = int(input[idx])
        idx += 1
        edges.append((u, v, w))
    
    min_cost = K  # Initialize with a value larger than any possible sum mod K
    
    # Iterate over all combinations of N-1 edges
    for combo in itertools.combinations(edges, N-1):
        # DSU to check connectivity
        parent = list(range(N + 1))  # 1-based indexing
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        def union(u, v):
            pu = find(u)
            pv = find(v)
            if pu != pv:
                parent[pv] = pu
        
        for u, v, w in combo:
            union(u, v)
        
        # Check if all nodes are connected
        root = find(1)
        all_connected = True
        for i in range(2, N + 1):
            if find(i) != root:
                all_connected = False
                break
        if all_connected:
            total = sum(w for u, v, w in combo) % K
            if total < min_cost:
                min_cost = total
    
    print(min_cost)

if __name__ == "__main__":
    main()