import sys
from itertools import combinations

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr += 1
    M = int(input[ptr]); ptr += 1
    K = int(input[ptr]); ptr += 1
    edges = []
    for _ in range(M):
        u = int(input[ptr]); ptr += 1
        v = int(input[ptr]); ptr += 1
        w = int(input[ptr]); ptr += 1
        edges.append((u, v, w))
    
    min_mod = K  # Initialize to a value larger than any possible mod
    target_edges = N - 1
    required_nodes = N
    
    for comb in combinations(edges, target_edges):
        # Build adjacency list
        adj = [[] for _ in range(N + 1)]
        for u, v, _ in comb:
            adj[u].append(v)
            adj[v].append(u)
        
        # DFS with bitmask
        visited = 1 << 0  # node 1 is bit 0
        stack = [1]
        count = 1
        
        while stack:
            node = stack.pop()
            for neighbor in adj[node]:
                bit = 1 << (neighbor - 1)
                if not (visited & bit):
                    visited |= bit
                    count += 1
                    stack.append(neighbor)
        
        if count == required_nodes:
            current_sum = 0
            for _, _, w in comb:
                current_sum += w
            current_mod = current_sum % K
            if current_mod < min_mod:
                min_mod = current_mod
                if min_mod == 0:
                    print(0)
                    return
    
    print(min_mod)

if __name__ == "__main__":
    main()