def main():
    import sys
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    
    # Read number of vertices and edges
    n = int(next(it))
    m = int(next(it))
    
    # Build DSU for original graph G.
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    # Union edges of graph G. Note that self-loops or multi-edges are possible.
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        union(u, v)
    
    # Process the K restrictions.
    k = int(next(it))
    # For each restriction, we store the pair of DSU leaders (in sorted order) corresponding to (x, y).
    forbidden = set()
    for _ in range(k):
        x = int(next(it)) - 1
        y = int(next(it)) - 1
        rx, ry = find(x), find(y)
        if rx > ry:
            rx, ry = ry, rx
        forbidden.add((rx, ry))
    
    # Process the Q queries.
    q = int(next(it))
    out_lines = []
    for _ in range(q):
        p = int(next(it)) - 1
        qv = int(next(it)) - 1
        rp, rq = find(p), find(qv)
        # If p and q are already in the same DSU component, adding an edge changes nothing.
        if rp == rq:
            out_lines.append("Yes")
        else:
            # Check if the extra edge would join a forbidden pair.
            a, b = rp, rq
            if a > b:
                a, b = b, a
            if (a, b) in forbidden:
                out_lines.append("No")
            else:
                out_lines.append("Yes")
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()