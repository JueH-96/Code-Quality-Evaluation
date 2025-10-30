def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    # We'll use a Union-Find (DSU) to compute the connected components of G.
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    def find(x):
        # Iterative path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx = find(x)
        ry = find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
    
    # Process the M edges of the graph G.
    # Note that self-loops or multi-edges don't affect connectivity.
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        union(u, v)
    
    # Process the K forbidden pairs.
    # In the given graph G these pairs are not connected.
    # We store each forbidden pair as an unordered pair of connected component roots.
    k = int(next(it))
    forbidden = set()
    for _ in range(k):
        x = int(next(it))
        y = int(next(it))
        rx, ry = find(x), find(y)
        # According to the problem, these must be in different components.
        if rx > ry:
            rx, ry = ry, rx
        forbidden.add((rx, ry))
    
    # Process the Q queries
    # For each query we add a new edge (p, q) to G.
    # If p and q are already in the same component, then nothing changes and G stays good.
    # Otherwise, the new edge merges two components: let A = find(p) and B = find(q).
    # The new graph becomes "bad" if there exists a forbidden pair that exactly connects A and B.
    q_queries = int(next(it))
    output = []
    for _ in range(q_queries):
        p = int(next(it))
        qv = int(next(it))
        rp, rq = find(p), find(qv)
        if rp == rq:
            output.append("Yes")
        else:
            a, b = rp, rq
            if a > b:
                a, b = b, a
            if (a, b) in forbidden:
                output.append("No")
            else:
                output.append("Yes")
    sys.stdout.write("
".join(output))
    
if __name__ == '__main__':
    main()