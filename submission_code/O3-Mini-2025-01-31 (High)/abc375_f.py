def main():
    import sys, numpy as np
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    q = int(next(it))
    
    # Read roads (each road given as A_i, B_i, C_i).
    # We convert cities to 0-indexed.
    roads = [None] * m
    for i in range(m):
        a = int(next(it)); b = int(next(it)); c = int(next(it))
        roads[i] = (a - 1, b - 1, c)
    
    # Process queries.
    # For queries of type "1 i" (road closures), mark the road as closed.
    # We also store all queries in order.
    queries = [None] * q
    closed = [False] * m
    for i in range(q):
        typ = next(it)
        if typ == b'1':
            # Query: "1 i" (road i becomes closed).
            road_index = int(next(it)) - 1
            queries[i] = (1, road_index)
            closed[road_index] = True
        else:
            # Query: "2 x y" (shortest path query)
            x = int(next(it)) - 1
            y = int(next(it)) - 1
            queries[i] = (2, x, y)
    
    # In the forward process the graph starts with all roads open.
    # Every road that is closed by any query will be absent in the final state.
    # So the "final state" graph contains all roads that were never closed.
    INF = 10**18
    d = np.full((n, n), INF, dtype=np.int64)
    np.fill_diagonal(d, 0)
    for i in range(m):
        if not closed[i]:
            u, v, w = roads[i]
            # Roads are bidirectional.
            if w < d[u, v]:
                d[u, v] = w
                d[v, u] = w
    
    # Compute the all‐pairs shortest paths for the final state using a vectorized Floyd–Warshall.
    for k in range(n):
        d = np.minimum(d, d[:, k:k+1] + d[k:k+1, :])
    
    # We now process the queries in reverse order.
    # In the forward process, a type 1 query permanently closes a road.
    # In reverse, that corresponds to “adding” the road back.
    # We will update our distance matrix "d" accordingly.
    answers = [None] * q  # We'll store answers for type2 queries.
    for i in range(q - 1, -1, -1):
        qu = queries[i]
        if qu[0] == 2:
            # Type 2 query: answer the shortest distance.
            _, x, y = qu
            res = int(d[x, y])
            if res >= INF:
                res = -1
            answers[i] = res
        else:
            # Type 1 query in forward order (road closed);
            # In reverse it means add that road back.
            _, road_index = qu
            u, v, w = roads[road_index]
            # Update the distance matrix:
            # For every pair (i, j) the new candidate routes are:
            #    d[i, u] + w + d[v, j]    and    d[i, v] + w + d[u, j]
            # We update d = min(d, candidate) in a fully vectorized fashion.
            cand = np.minimum(d[:, u:u+1] + w + d[v:v+1, :],
                              d[:, v:v+1] + w + d[u:u+1, :])
            d = np.minimum(d, cand)
    
    # Output the answers for type2 queries in forward order.
    out_lines = []
    for i in range(q):
        if queries[i][0] == 2:
            out_lines.append(str(answers[i]))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()