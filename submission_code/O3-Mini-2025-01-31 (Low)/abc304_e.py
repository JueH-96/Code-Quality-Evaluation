def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # DSU for connectivity in the original graph G
    parent = list(range(n + 1))
    rank = [0] * (n + 1)
    
    def find(x):
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
    
    # Process M edges of original graph G (self-loops and multi-edges are allowed)
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        union(u, v)
    
    # Process K forbidden pairs; each forbidden pair (x,y) means there is no path
    # in G connecting x and y; thus x and y lie in different connected components.
    # In the new graph G^(i), if we add an edge that connects the two corresponding
    # DSU components then the forbidden condition gets violated.
    K = int(next(it))
    forbidden = set()
    for _ in range(K):
        x = int(next(it))
        y = int(next(it))
        rx = find(x)
        ry = find(y)
        if rx > ry:
            rx, ry = ry, rx
        forbidden.add((rx, ry))
    
    # Process Q queries. For each query, we add an edge (p, q):
    # If p and q are in the same DSU component, new edge won't change connectivity,
    # so answer "Yes".
    # If different, then check if these two DSU roots create a forbidden pair.
    Q = int(next(it))
    result = []
    for _ in range(Q):
        p = int(next(it))
        q = int(next(it))
        rp = find(p)
        rq = find(q)
        if rp == rq:
            result.append("Yes")
        else:
            comp_pair = (rp, rq) if rp < rq else (rq, rp)
            if comp_pair in forbidden:
                result.append("No")
            else:
                result.append("Yes")
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()