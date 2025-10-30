def main():
    import sys
    import numpy as np
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    q = int(next(it))
    
    roads = []
    for _ in range(m):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        c = int(next(it))
        roads.append((a, b, c))
    
    # Read queries.
    # Also mark which roads get removed (closed permanently in forward time).
    queries = []
    road_removed = [False] * m
    for _ in range(q):
        typ = next(it)
        if typ == "1":
            idx = int(next(it)) - 1
            queries.append(("removal", idx))
            road_removed[idx] = True
        else:
            x = int(next(it)) - 1
            y = int(next(it)) - 1
            queries.append(("query", x, y))
    
    INF = 10**18
    # Build final graph distance matrix: only include roads that were never removed.
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for i, (a, b, c) in enumerate(roads):
        if not road_removed[i]:
            if c < dist[a][b]:
                dist[a][b] = c
                dist[b][a] = c

    # Run Floyd Warshall on the final graph
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            di = dist[i]
            alt = di[k]
            if alt == INF:
                continue
            for j in range(n):
                newd = alt + dk[j]
                if newd < di[j]:
                    di[j] = newd

    # Convert distance matrix into a NumPy array.
    d = np.array(dist, dtype=np.int64)
    INF_np = 10**18

    # Process queries in reverse order.
    # For a query type "2 x y" (forward query) we simply record the current distance.
    # For "1 i" (i.e. removal in forward order), we add that road in reverse order.
    answers = []
    for op in reversed(queries):
        if op[0] == "query":
            x, y = op[1], op[2]
            res = int(d[x, y])
            answers.append(-1 if res >= INF_np else res)
        else:
            i = op[1]
            u, v, w = roads[i]
            # The new edge gives a possible improvement: 
            # For any pair (i, j) we can try: d[i][j] = min( d[i][j], d[i][u] + w + d[v][j], d[i][v] + w + d[u][j] ).
            cand1 = d[:, u:u+1] + w + d[v:v+1, :]
            cand2 = d[:, v:v+1] + w + d[u:u+1, :]
            # Update all distances
            update = np.minimum(cand1, cand2)
            d = np.minimum(d, update)
            # Because using the new edge may allow further improvements via u or v, we do an extra relax step.
            for k in (u, v):
                d = np.minimum(d, d[:, k:k+1] + d[k:k+1, :])
    
    # Reverse answers to print them in the original order.
    out_lines = "
".join(str(ans) for ans in reversed(answers))
    sys.stdout.write(out_lines)

if __name__ == '__main__':
    main()