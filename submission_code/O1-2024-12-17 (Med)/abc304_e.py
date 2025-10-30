def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin.read().strip().split()
    # Read N (vertices) and M (edges)
    N = int(data[0])
    M = int(data[1])
    idx = 2

    # DSU (Union-Find) setup for the original graph G
    parent = list(range(N))
    size = [1]*N
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

    # Read and union all edges
    for _ in range(M):
        u = int(data[idx]) - 1
        v = int(data[idx+1]) - 1
        idx += 2
        if u != v:
            union(u, v)

    # Read K and the K forbidden pairs (which must remain disconnected)
    K = int(data[idx])
    idx += 1

    # We only need to store, for each leader, which other leaders it cannot merge with
    forbidden = {}

    for _ in range(K):
        x = int(data[idx]) - 1
        y = int(data[idx+1]) - 1
        idx += 2
        lx = find(x)
        ly = find(y)
        # Since G is "good", lx != ly is guaranteed by the problem statement
        if lx not in forbidden:
            forbidden[lx] = set()
        forbidden[lx].add(ly)
        if ly not in forbidden:
            forbidden[ly] = set()
        forbidden[ly].add(lx)

    # Read Q and process each query independently
    Q = int(data[idx])
    idx += 1

    out = []
    for _ in range(Q):
        p = int(data[idx]) - 1
        q = int(data[idx+1]) - 1
        idx += 2
        lp = find(p)
        lq = find(q)
        # If they are already in the same component, adding the edge changes nothing => "Yes"
        if lp == lq:
            out.append("Yes")
        else:
            # If there is a forbidden pair (lp, lq), then merging them would violate goodness
            if lp in forbidden and lq in forbidden[lp]:
                out.append("No")
            else:
                out.append("Yes")

    print("
".join(out))

# Don't forget to call main()
if __name__ == "__main__":
    main()