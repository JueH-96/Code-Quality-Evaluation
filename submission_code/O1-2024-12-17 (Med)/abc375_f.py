def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast iterator over input_data
    # because Q can be large, we avoid repeated function calls to input()
    it = 0
    def read_int():
        nonlocal it
        val = int(input_data[it])
        it += 1
        return val

    INF = 10**15
    
    N = read_int()
    M = read_int()
    Q = read_int()
    
    # Store roads so we can "close" them later by index
    roads = []
    for _ in range(M):
        A = read_int()
        B = read_int()
        C = read_int()
        roads.append((A-1, B-1, C))  # zero-based internally
    
    # Adjacency matrix for direct edges (will set to INF when an edge is closed)
    # Initialize with INF except 0 on diagonal
    adjacency = [[INF]*N for _ in range(N)]
    for i in range(N):
        adjacency[i][i] = 0
    
    # Fill adjacency for existing roads
    for (u,v,c) in roads:
        adjacency[u][v] = min(adjacency[u][v], c)
        adjacency[v][u] = min(adjacency[v][u], c)
    
    # dist will hold the all-pairs shortest paths for the current state
    dist = [[adjacency[i][j] for j in range(N)] for i in range(N)]
    
    # Floyd-Warshall to get initial all-pairs distances
    for k in range(N):
        dk = dist[k]
        for i in range(N):
            di = dist[i]
            ik = di[k]
            if ik == INF:
                continue
            for j in range(N):
                alt = ik + dk[j]
                if alt < di[j]:
                    di[j] = alt
    
    # We process queries in order; each time we see a type-1 (close) query,
    # we remove that edge from adjacency and recalc the full Floyd-Warshall.
    
    out = []
    for _ in range(Q):
        t = read_int()
        if t == 1:
            # close a road
            i = read_int() - 1
            (u,v,c) = roads[i]
            if adjacency[u][v] < INF:  # not already closed
                # remove from adjacency
                adjacency[u][v] = INF
                adjacency[v][u] = INF
                # Recompute dist from scratch using updated adjacency
                for r in range(N):
                    for s in range(N):
                        dist[r][s] = adjacency[r][s]
                # Floyd-Warshall again
                for k in range(N):
                    dk = dist[k]
                    for p in range(N):
                        dp = dist[p]
                        pk = dp[k]
                        if pk == INF:
                            continue
                        for q in range(N):
                            alt = pk + dk[q]
                            if alt < dp[q]:
                                dp[q] = alt
        else:
            # t == 2, distance query
            x = read_int() - 1
            y = read_int() - 1
            dxy = dist[x][y]
            out.append(str(dxy if dxy < INF else -1))
    
    print("
".join(out))

# Call main() as required
if __name__ == "__main__":
    main()