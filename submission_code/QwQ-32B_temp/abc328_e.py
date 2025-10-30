import sys
from itertools import combinations

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        edges.append((u, v, w))
    
    min_mod = K  # Initialize to a value larger than possible mod (since mod is < K)
    
    for combo in combinations(edges, N-1):
        parent = list(range(N + 1))  # 1-based
        
        # Process unions with path compression
        for u, v, w in combo:
            # Find root of u
            pu = u
            while parent[pu] != pu:
                parent[pu] = parent[parent[pu]]  # Path compression
                pu = parent[pu]
            # Find root of v
            pv = v
            while parent[pv] != pv:
                parent[pv] = parent[parent[pv]]
                pv = parent[pv]
            if pu != pv:
                parent[pu] = pv
        
        # Check connectivity
        # Find root of node 1
        root = 1
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        
        connected = True
        for node in range(2, N+1):
            current = node
            while parent[current] != current:
                parent[current] = parent[parent[current]]
                current = parent[current]
            if current != root:
                connected = False
                break
        
        if connected:
            total = sum(w for u, v, w in combo)
            mod_val = total % K
            if mod_val < min_mod:
                min_mod = mod_val
    
    print(min_mod)

if __name__ == "__main__":
    main()