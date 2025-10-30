def main():
    import sys
    import heapq
    sys.setrecursionlimit(10**7)
    
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    edges = []
    idx = 2
    
    # Read edges
    adjacency = [[] for _ in range(N+1)]
    for i in range(M):
        A = int(input_data[idx]); B = int(input_data[idx+1]); C = int(input_data[idx+2])
        idx += 3
        edges.append((A, B, C))
        # Store for Dijkstra
        adjacency[A].append((B, C, i))
        adjacency[B].append((A, C, i))

    INF = 10**18
    # Dijkstra from city 1
    d1 = [INF]*(N+1)
    d1[1] = 0
    pq = [(0, 1)]
    while pq:
        cd, u = heapq.heappop(pq)
        if cd > d1[u]:
            continue
        for v, cost, _ in adjacency[u]:
            nd = cd + cost
            if nd < d1[v]:
                d1[v] = nd
                heapq.heappush(pq, (nd, v))

    # Dijkstra from city N
    dN = [INF]*(N+1)
    dN[N] = 0
    pq = [(0, N)]
    while pq:
        cd, u = heapq.heappop(pq)
        if cd > dN[u]:
            continue
        for v, cost, _ in adjacency[u]:
            nd = cd + cost
            if nd < dN[v]:
                dN[v] = nd
                heapq.heappush(pq, (nd, v))

    dist_all = d1[N]  # Shortest distance 1->N with all roads

    # Build "shortest-path subgraph" G' (directed) where each edge
    # is used in at least one shortest path 1->N
    used = [False]*M
    out_edges = [[] for _ in range(N+1)]
    for i, (u, v, c) in enumerate(edges):
        # Check if this edge can appear on a shortest path 1->N
        # Condition: d1[u] + c + dN[v] == d1[N], and orientation from smaller d1 to bigger d1
        if d1[u] + c + dN[v] == dist_all and d1[u] < d1[v]:
            used[i] = True
            out_edges[u].append((v, i))
        elif d1[v] + c + dN[u] == dist_all and d1[v] < d1[u]:
            used[i] = True
            out_edges[v].append((u, i))
        # else not in G'

    # We will count the number of shortest paths from 1 to each node (mod two large primes)
    p1, p2 = 10**9+7, 10**9+9
    
    ways = [[0, 0] for _ in range(N+1)]
    ways[1] = [1, 1]  # 1 way to reach city 1 from 1
    
    # Sort nodes by d1 (ascending) to do a DAG-like DP
    nodes_asc = list(range(1, N+1))
    nodes_asc.sort(key=lambda x: d1[x])
    
    for u in nodes_asc:
        w1, w2 = ways[u]
        if w1 == 0 and w2 == 0:
            continue
        for (nx, _) in out_edges[u]:
            ways[nx][0] = (ways[nx][0] + w1) % p1
            ways[nx][1] = (ways[nx][1] + w2) % p2

    # ways[N] is the total number of shortest paths (mod p1, p2)
    ways_1_to_N = (ways[N][0], ways[N][1])

    # Now count ways backward: number of ways from each node to N in G'
    # We'll build the "in-edges" of G'
    in_edges = [[] for _ in range(N+1)]
    for u in range(1, N+1):
        for (v, eidx) in out_edges[u]:
            in_edges[v].append((u, eidx))

    ways_bwd = [[0, 0] for _ in range(N+1)]
    ways_bwd[N] = [1, 1]  # 1 way to reach N from N
    
    # Sort nodes by d1 descending for backward DP
    nodes_desc = list(range(1, N+1))
    nodes_desc.sort(key=lambda x: d1[x], reverse=True)
    
    for u in nodes_desc:
        wb1, wb2 = ways_bwd[u]
        if wb1 == 0 and wb2 == 0:
            continue
        for (prevu, _) in in_edges[u]:
            ways_bwd[prevu][0] = (ways_bwd[prevu][0] + wb1) % p1
            ways_bwd[prevu][1] = (ways_bwd[prevu][1] + wb2) % p2

    # Finally, for each edge i in [1..M], decide "Yes" or "No"
    # If not used[i], answer is "No" because it does not affect the shortest path.
    # If used[i], let the edge be oriented (u->v). Then it is on all shortest paths
    # iff ways[u] * ways_bwd[v] == ways[N] (in both moduli).
    
    ans = []
    wt1, wt2 = ways_1_to_N
    for i, (u, v, c) in enumerate(edges):
        if not used[i]:
            ans.append("No")
            continue
        # Determine orientation
        if d1[u] < d1[v]:
            su1, su2 = ways[u]
            tv1, tv2 = ways_bwd[v]
        else:
            su1, su2 = ways[v]
            tv1, tv2 = ways_bwd[u]
        prod1 = (su1 * tv1) % p1
        prod2 = (su2 * tv2) % p2
        if prod1 == wt1 and prod2 == wt2:
            ans.append("Yes")
        else:
            ans.append("No")

    print("
".join(ans))

# Do not forget to call main()
main()