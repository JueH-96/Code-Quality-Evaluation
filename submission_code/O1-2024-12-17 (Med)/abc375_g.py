def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    # Faster iterator-style input parsing
    it = 0
    def read():
        nonlocal it
        val = input_data[it]
        it += 1
        return val

    N = int(read())
    M = int(read())
    edges = []
    adj_for_1 = [[] for _ in range(N+1)]  # adjacency for running dijkstra from node 1
    adj_for_N = [[] for _ in range(N+1)]  # adjacency for running dijkstra from node N
    
    for i in range(M):
        A = int(read())
        B = int(read())
        C = int(read())
        edges.append((A,B,C))
        adj_for_1[A].append((B,C))
        adj_for_1[B].append((A,C))
        # for dijkstra from N, just reverse perspective (same undirected edges)
        adj_for_N[B].append((A,C))
        adj_for_N[A].append((B,C))

    # Dijkstra to get dist1 (distance from city 1)
    INF = 10**18

    def dijkstra(start, graph):
        dist = [INF]*(N+1)
        dist[start] = 0
        pq = [(0, start)]
        while pq:
            cd, u = heapq.heappop(pq)
            if cd > dist[u]:
                continue
            for (v, w) in graph[u]:
                nd = cd + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    dist1 = dijkstra(1, adj_for_1)
    distN = dijkstra(N, adj_for_N)
    
    dist_all = dist1[N]  # shortest distance from 1 to N using all roads
    
    # Build "direction" array to record if an edge is used in the DAG of shortest paths
    # direction[i] = (u,v) means edge i is directed u->v in the shortest-path DAG
    # direction[i] = None means the edge is not in any shortest-path DAG
    direction = [None]*M
    for i in range(M):
        u, v, c = edges[i]
        # Check if we can direct u->v
        # We need dist1[u] + c + distN[v] == dist_all AND dist1[u] + c == dist1[v]
        if dist1[u] + c + distN[v] == dist_all and dist1[u] + c == dist1[v]:
            direction[i] = (u,v)
        # Else check if we can direct v->u
        elif dist1[v] + c + distN[u] == dist_all and dist1[v] + c == dist1[u]:
            direction[i] = (v,u)
        # Otherwise, direction[i] stays None

    # Build forward adjacency (DAG) and backward adjacency
    fwd = [[] for _ in range(N+1)]
    bwd = [[] for _ in range(N+1)]
    for i in range(M):
        if direction[i] is not None:
            x, y = direction[i]
            fwd[x].append(y)
            bwd[y].append(x)

    # We will compute dp_in[u] = number of ways (modded) to reach u from 1 in the DAG
    # and dp_out[u] = number of ways (modded) to reach N from u in the DAG.
    # Use Two-Mod technique to avoid collisions in equality checks.

    MOD1 = 10**9+7
    MOD2 = 10**9+9

    dp_in1 = [0]*(N+1)
    dp_in2 = [0]*(N+1)
    dp_out1 = [0]*(N+1)
    dp_out2 = [0]*(N+1)

    # Sort nodes by dist1 ascending (no ties for edges used in shortest paths with positive costs)
    nodes = list(range(1, N+1))
    nodes.sort(key=lambda x: dist1[x])

    # dp_in from 1
    dp_in1[1] = 1
    dp_in2[1] = 1
    for u in nodes:
        # forward edges from u -> v
        in1 = dp_in1[u]
        in2 = dp_in2[u]
        if (in1 == 0 and in2 == 0):
            continue
        for v in fwd[u]:
            dp_in1[v] = (dp_in1[v] + in1) % MOD1
            dp_in2[v] = (dp_in2[v] + in2) % MOD2

    # dp_out from N
    dp_out1[N] = 1
    dp_out2[N] = 1
    # sort nodes by dist1 descending
    nodes_desc = nodes[::-1]
    for u in nodes_desc:
        out1 = dp_out1[u]
        out2 = dp_out2[u]
        if (out1 == 0 and out2 == 0):
            continue
        for v in bwd[u]:  # v -> u in DAG
            dp_out1[v] = (dp_out1[v] + out1) % MOD1
            dp_out2[v] = (dp_out2[v] + out2) % MOD2

    # ways_total = dp_in( N )
    ways_total_1 = dp_in1[N]
    ways_total_2 = dp_in2[N]

    # Utility to multiply under mod
    def eq_product(u,v):
        # we check if dp_in[u]*dp_out[v] == dp_in[N] (in both mods)
        # i.e. (dp_in1[u]*dp_out1[v] mod MOD1 == ways_total_1) and likewise for MOD2
        return ((dp_in1[u]*dp_out1[v]) % MOD1 == ways_total_1) and \
               ((dp_in2[u]*dp_out2[v]) % MOD2 == ways_total_2)

    out = []
    for i in range(M):
        dire = direction[i]
        if dire is None:
            # Edge not in any shortest path => removing it won't affect shortest distance
            out.append("No")
        else:
            u, v = dire
            # Check if dp_in[u]*dp_out[v] == dp_in[N] in our double-mod sense
            if eq_product(u, v):
                # This edge is used by all shortest paths
                out.append("Yes")
            else:
                out.append("No")

    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()