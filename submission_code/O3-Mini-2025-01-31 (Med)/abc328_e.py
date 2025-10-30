def main():
    import sys, itertools
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    K = int(next(it))
    edges = []
    for _ in range(m):
        u = int(next(it)) - 1  # convert to 0-indexed
        v = int(next(it)) - 1
        w = int(next(it))
        edges.append((u, v, w))
        
    best = K  # mod cost is in the range [0, K-1], so best initializing with K is safe.
    
    # Each spanning tree will be an (n-1)-combination of edges.
    # Since n <= 8 and m <= n(n-1)/2 (max 28 edges), the total number of combinations
    # is sufficiently small to try all.
    for comb in itertools.combinations(range(m), n - 1):
        parent = list(range(n))
        valid = True
        total = 0

        # Inline union-find (with iterative find using path compression)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for idx in comb:
            u, v, w = edges[idx]
            ru = find(u)
            rv = find(v)
            if ru == rv:
                valid = False
                break
            parent[rv] = ru
            total += w
            
        # If this combination of edges forms a spanning tree,
        # consider its cost modulo K.
        if valid:
            cost_mod = total % K
            if cost_mod < best:
                best = cost_mod
                # If we ever get a cost of 0, it's the minimum possible so we can exit early.
                if best == 0:
                    print(0)
                    return
    print(best)
    
if __name__ == '__main__':
    main()