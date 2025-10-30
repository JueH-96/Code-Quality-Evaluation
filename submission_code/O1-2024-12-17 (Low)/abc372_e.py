def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast IO helpers:
    # input_data is a list of strings; we'll pop from the left as we parse.
    # This avoids calling input() repeatedly.

    # Disjoint Set Union (Union-Find) with tracking of top-10 largest vertices per component
    sys.setrecursionlimit(10**7)

    # Parse N, Q
    N = int(input_data[0])
    Q = int(input_data[1])

    idx = 2  # index in input_data

    parent = list(range(N+1))
    size = [1] * (N+1)
    # For each component's root, store up to 10 largest vertices (descending)
    top10 = [[] for _ in range(N+1)]
    # Initialize top10 for each vertex (each is alone)
    for v in range(1, N+1):
        top10[v].append(v)

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        # Union by size
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        # Now ra is the bigger root
        parent[rb] = ra
        size[ra] += size[rb]
        # Merge top10[rb] into top10[ra]
        # We only need the top 10 in descending order
        merged = []
        # both lists are already in descending order
        i, j = 0, 0
        while (i < len(top10[ra]) or j < len(top10[rb])) and len(merged) < 10:
            val_ra = top10[ra][i] if i < len(top10[ra]) else -1
            val_rb = top10[rb][j] if j < len(top10[rb]) else -1
            if val_ra >= val_rb:
                merged.append(val_ra)
                i += 1
            else:
                merged.append(val_rb)
                j += 1
        top10[ra] = merged
        top10[rb].clear()  # not strictly necessary but can save memory

    out = []
    # Process queries
    ptr = 0
    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        if t == 1:
            # 1 u v
            u = int(input_data[idx]); idx += 1
            v = int(input_data[idx]); idx += 1
            union(u, v)
        else:
            # 2 v k
            v = int(input_data[idx]); idx += 1
            k = int(input_data[idx]); idx += 1
            rv = find(v)
            if k <= len(top10[rv]):
                out.append(str(top10[rv][k-1]))
            else:
                out.append(str(-1))

    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()