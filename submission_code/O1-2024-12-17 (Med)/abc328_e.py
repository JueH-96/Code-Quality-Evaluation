def main():
    import sys
    from itertools import combinations

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    edges = []
    idx = 3
    for _ in range(M):
        u = int(input_data[idx]); v = int(input_data[idx+1]); w = int(input_data[idx+2])
        idx += 3
        # Store edges in zero-based indexing for convenience in union-find
        edges.append((u-1, v-1, w))
    
    # A small union-find / disjoint-set implementation
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b, parent, rank):
        if rank[a] < rank[b]:
            parent[a] = b
        elif rank[a] > rank[b]:
            parent[b] = a
        else:
            parent[b] = a
            rank[a] += 1

    min_cost = K  # Since valid costs mod K lie in [0, K-1], use K as an initial "infinity"

    # Check all combinations of size N-1
    for comb in combinations(range(M), N-1):
        # Initialize union-find
        parent = list(range(N))
        rank = [0]*N
        merges = 0
        has_cycle = False
        total_weight = 0
        
        for e_idx in comb:
            u, v, w = edges[e_idx]
            ru = find(u, parent)
            rv = find(v, parent)
            if ru == rv:
                has_cycle = True
                break
            union(ru, rv, parent, rank)
            merges += 1
            total_weight += w
        
        if not has_cycle and merges == N-1:
            cost_mod = total_weight % K
            if cost_mod < min_cost:
                min_cost = cost_mod
        
        # Early exit if we ever find a spanning tree with cost_mod == 0
        if min_cost == 0:
            break

    print(min_cost)

# IMPORTANT: Do not forget to call main()
main()