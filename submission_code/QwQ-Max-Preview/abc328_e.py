import itertools

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    m = int(input[idx])
    idx += 1
    k = int(input[idx])
    idx += 1
    edges = []
    for _ in range(m):
        u = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        w = int(input[idx])
        idx += 1
        edges.append((u, v, w))
    
    min_mod = k  # Initialize with maximum possible mod value (K, since mod K can't exceed K-1)
    
    for combo in itertools.combinations(edges, n-1):
        # Check if this combination forms a spanning tree
        parent = list(range(n + 1))  # 1-based indexing for nodes
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        for u, v, _ in combo:
            pu = find(u)
            pv = find(v)
            if pu != pv:
                parent[pv] = pu
        
        # Check if all nodes are connected
        root = find(1)
        connected = True
        for node in range(2, n + 1):
            if find(node) != root:
                connected = False
                break
        if connected:
            total = sum(w for _, _, w in combo)
            current_mod = total % k
            if current_mod < min_mod:
                min_mod = current_mod
    
    print(min_mod)

if __name__ == "__main__":
    main()