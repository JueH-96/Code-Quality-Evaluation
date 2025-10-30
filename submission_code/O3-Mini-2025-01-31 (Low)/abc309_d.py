def main():
    import sys
    from collections import deque
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    
    # Read the input
    N1 = int(next(it))
    N2 = int(next(it))
    M = int(next(it))
    total = N1 + N2

    # Build separate graphs for the two clusters:
    #  - Cluster 1: vertices 1..N1 (indexed 1..N1)
    #  - Cluster 2: vertices N1+1..total, we store them as 1..N2 by subtracting N1.
    g1 = [[] for _ in range(N1 + 1)]
    g2 = [[] for _ in range(N2 + 1)]
    
    # Process the M edges. We only add an edge to a cluster if both endpoints lie in the same cluster.
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        if u <= N1 and v <= N1:
            # both vertices in cluster 1
            g1[u].append(v)
            g1[v].append(u)
        elif u > N1 and v > N1:
            # both vertices in cluster 2 -- shift indices by N1
            uu = u - N1
            vv = v - N1
            g2[uu].append(vv)
            g2[vv].append(uu)
        # If the edge connects a vertex from cluster1 and a vertex from cluster2, we ignore it.
        # This is because by problem guarantee, 1 and total (N1+N2) are disconnected in the original graph,
        # and such edges would not appear.

    # For the edge we add between clusters, note that the new shortest path will always use that edge.
    # In cluster 1, from vertex 1 to some u, the distance is fixed internally.
    # In cluster 2, from some vertex v to vertex (N1+N2), the distance is fixed internally if we shift the labeling.
    #
    # So if we add an edge between vertex u (in cluster 1) and vertex v (in cluster 2),
    # the distance between 1 and (N1+N2) becomes:
    #     d = (distance from 1 to u in cluster 1) + 1 + (distance from v to N1+N2 in cluster 2)
    #
    # To maximize d, we simply choose the u in cluster 1 with the maximum distance from 1,
    # and the v in cluster 2 with the maximum distance from (N1+N2).
    
    # Compute the distances from vertex 1 in cluster1 (vertices 1..N1)
    d1 = [-1] * (N1 + 1)
    dq = deque()
    d1[1] = 0
    dq.append(1)
    while dq:
        cur = dq.popleft()
        for nxt in g1[cur]:
            if d1[nxt] == -1:
                d1[nxt] = d1[cur] + 1
                dq.append(nxt)
    max_d1 = max(d1[1:])  # Cluster1 is connected by guarantee.

    # Compute the distances in cluster2 from vertex (N1+N2).
    # In our g2, vertex (N1+N2) is represented as vertex (total - N1)
    start_cluster2 = total - N1
    d2 = [-1] * (N2 + 1)
    dq = deque()
    d2[start_cluster2] = 0
    dq.append(start_cluster2)
    while dq:
        cur = dq.popleft()
        for nxt in g2[cur]:
            if d2[nxt] == -1:
                d2[nxt] = d2[cur] + 1
                dq.append(nxt)
    max_d2 = max(d2[1:])  # Cluster2 is also connected by guarantee.

    # The maximum possible shortest path length after adding one edge is:
    #      max_d1 + 1 + max_d2
    ans = max_d1 + 1 + max_d2
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()